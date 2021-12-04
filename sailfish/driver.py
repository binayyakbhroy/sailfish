import os, pickle, pathlib, logging
from typing import NamedTuple, Dict
from sailfish.task import Recurrence, RecurringTask, ParseRecurrenceError
from sailfish.setup import Setup, SetupError

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """An invalid runtime configuration"""


def first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg


def update_dict_if_none(new_dict, old_dict, frozen=[]):
    """
    Like `dict.update`, except `key=value` pairs in `old_dict` are only used
    to add / overwrite values in `new_dict` if they are `None` or missing.
    """
    for key in old_dict:
        old_val = old_dict.get(key)
        new_val = new_dict.get(key)

        if type(new_val) is dict and type(old_val) is dict:
            update_dict_if_none(new_val, old_val)

        elif old_val is not None:
            if new_val is None:
                new_dict[key] = old_val
            elif key in frozen and new_val != old_val:
                raise ConfigurationError(f"{key} cannot be changed")


def update_if_none(new, old, frozen=[]):
    """
    Same as `update_dict_if_none`, except operates on (immutable) named tuple
    instances and returns a new named tuple.
    """
    new_dict = new._asdict()
    old_dict = old._asdict()
    update_dict_if_none(new_dict, old_dict, frozen)
    return type(new)(**new_dict)


def asdict(t):
    """
    Convert named tuple instances to dictionaries.

    This function operates recursively on the data members of a dictionary or
    named tuple. Each object that is a named tuple is mapped to its dictionary
    representation, with an additional `_type` key to indicate the named tuple
    subclass. This mapping is applied to the simulation state before pickling,
    so that `sailfish` module is not required to unpickle the checkpoint
    files.
    """
    if type(t) is dict:
        return {k: asdict(v) for k, v in t.items()}
    if isinstance(t, tuple):
        d = {k: asdict(v) for k, v in t._asdict().items()}
        d["_type"] = type(t).__name__
        return d
    return t


def fromdict(d):
    """
    Convert from dictionaries to named tuples.

    This function performs the inverse of the `asdict` method, and is applied
    to pickled simulation states.
    """
    if type(d) is dict:
        if "_type" in d:
            cls = eval(d["_type"])
            del d["_type"]
            return cls(**d)
        else:
            return {k: fromdict(v) for k, v in d.items()}
    else:
        return d


def write_checkpoint(number, outdir, state):
    """
    Write the simulation state to a file, as a pickle.
    """
    if type(number) is int:
        filename = f"chkpt.{number:04d}.pk"
    else:
        filename = f"chkpt.final.pk"

    if outdir is not None:
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outdir, filename)

    with open(filename, "wb") as chkpt:
        if logger is not None:
            logger.info(f"write checkpoint {chkpt.name}")
        pickle.dump(asdict(state), chkpt)


def load_checkpoint(chkpt_file):
    """
    Load the simulation state from a pickle file.
    """
    try:
        with open(chkpt_file, "rb") as file:
            return fromdict(pickle.load(file))
    except FileNotFoundError:
        raise ConfigurationError(f"could not open checkpoint file {chkpt_file}")


def initial_condition(setup, num_zones, domain):
    import numpy as np

    xcells = np.linspace(domain[0], domain[1], num_zones)
    primitive = np.zeros([num_zones, 4])

    for x, p in zip(xcells, primitive):
        setup.initial_primitive(x, p)

    return primitive


class DriverArgs(NamedTuple):
    """
    Contains data used by the driver.
    """

    setup_name: str = None
    chkpt_file: str = None
    model_parameters: dict = None
    cfl_number: float = None
    end_time: float = None
    execution_mode: str = None
    fold: int = None
    resolution: int = None
    events: Dict[str, Recurrence] = dict()

    def from_namespace(args):
        """
        Construct an instance from an argparse-type namespace object.
        """

        def parse_item(item):
            try:
                key, val = item.split("=")
                return key, eval(val)
            except (NameError, ValueError):
                raise ConfigurationError(f"badly formed model parameter {item}")

        driver = DriverArgs(
            **{k: w for k, w in vars(args).items() if k in DriverArgs._fields}
        )
        parts = args.command.split(":")

        if not parts[0].endswith(".pk"):
            setup_name = parts[0]
            chkpt_file = None
        else:
            setup_name = None
            chkpt_file = parts[0]

        try:
            model_parameters = dict(parse_item(a) for a in parts[1:])
        except IndexError:
            model_parameters = dict()

        return driver._replace(
            setup_name=setup_name,
            chkpt_file=chkpt_file,
            model_parameters=model_parameters,
        )

    def updated_parameter_dict(self, old, setup_cls):
        """
        Return an updated parameter dict.

        The old parameter dict is expected to be read from a checkpoint file.
        The setup class is used to determine which of the parameter entries in
        the new and old parameter files need to match, or not be superseded.
        """
        new = self.model_parameters
        update_dict_if_none(new, old, frozen=list(setup_cls.immutable_parameter_keys))
        return new

    @property
    def fresh_setup(self):
        """
        Return true if this is not a restart.
        """
        return self.setup_name is not None


def simulate(driver):
    """
    Main denerator for running simulations.

    If invoked with a `DriverArgs` instance in `driver`, the other arguments
    are ignored. Otherwise, the driver is created from the setup name, model
    paramters, and keyword arguments.

    This function is a generator: it yields its state at a sequence of
    pause points, defined by the `events` dictionary.
    """

    from time import perf_counter
    from logging import getLogger, basicConfig, INFO, StreamHandler, Formatter
    from sailfish import system
    from sailfish.setup import Setup
    from sailfish.solvers import srhd_1d
    from sailfish.task import Recurrence

    """
    Initialize and log state in the the system module. The build system
    influences JIT-compiled module code. Currently the build parameters are
    inferred from the platform (Linux or MacOS), but in the future these
    should also be extensible by a system-specific rc-style configuration
    file.
    """
    system.configure_build()
    system.log_system_info(driver.execution_mode or "cpu")
    loop_logger = getLogger("loop_message")

    if driver.fresh_setup:
        """
        Generate an initial driver state from command line arguments, model
        parametrs, and a setup instance.
        """
        logger.info(f"generate initial data for setup {driver.setup_name}")
        setup = Setup.find_setup_class(driver.setup_name)(
            **driver.model_parameters or dict()
        )
        driver = driver._replace(
            resolution=driver.resolution or setup.default_resolution,
        )

        iteration = 0
        time = 0.0
        tasks = {name: RecurringTask() for name in driver.events}
        initial = initial_condition(setup, driver.resolution, setup.domain)
    else:
        """
        Load driver state from a checkpoint file. The setup model parameters
        are updated with any items given on the command line after the setup
        name. All command line arguments are also restorted from the
        previous session, but are updated with the command line argument
        given for this session, except for "frozen" arguments.
        """
        logger.info(f"load checkpoint {driver.chkpt_file}")
        chkpt = load_checkpoint(driver.chkpt_file)
        setup_cls = Setup.find_setup_class(chkpt["setup_name"])
        driver = update_if_none(driver, chkpt["driver"], frozen=["resolution"])
        params = driver.updated_parameter_dict(chkpt["parameters"], setup_cls)
        setup = setup_cls(**params)

        iteration = chkpt["iteration"]
        time = chkpt["time"]
        tasks = chkpt["tasks"]
        initial = chkpt["primitive"]

        for event in driver.events:
            if event not in tasks:
                tasks[event] = RecurringTask()

    mode = driver.execution_mode or "cpu"
    fold = driver.fold or 10
    cfl_number = driver.cfl_number or 0.6
    dx = (setup.domain[1] - setup.domain[0]) / driver.resolution
    dt = dx * cfl_number
    end_time = first_not_none(driver.end_time, setup.default_end_time, float("inf"))

    # Construct a solver instance. TODO: the solver should be obtained from
    # the setup instance.
    solver = srhd_1d.Solver(
        initial=initial,
        time=time,
        domain=setup.domain,
        num_patches=1,
        boundary_condition=setup.boundary_condition,
        mode=mode,
    )

    for name, event in driver.events.items():
        logger.info(f"recurrence for {name} event is {event}")

    logger.info(f"run until t={end_time}")
    logger.info(f"CFL number is {cfl_number}")
    logger.info(f"timestep is {dt}")
    setup.print_model_parameters(newlines=True)

    def grab_state():
        """
        Collect items from the driver and solver state, as well as run
        details, sufficient for restarts and post processing.
        """
        return dict(
            iteration=iteration,
            time=solver.time,
            primitive=solver.primitive,
            tasks=tasks,
            driver=driver,
            parameters=setup.model_parameter_dict,
            setup_name=setup.dash_case_class_name,
            domain=setup.domain,
        )

    while end_time is None or end_time > solver.time:
        """
        Run the main simulation loop. Iterations are grouped according the
        the fold parameter. Side effects including the iteration message are
        performed between fold boundaries.
        """

        for name, event in driver.events.items():
            task = tasks[name]
            if tasks[name].is_due(solver.time, event):
                tasks[name] = task.next(solver.time, event)
                yield name, task.number, grab_state()

        with system.measure_time() as fold_time:
            for _ in range(fold):
                solver.new_timestep()
                solver.advance_rk(0.0, dt)
                solver.advance_rk(0.5, dt)
                iteration += 1

        Mzps = driver.resolution / fold_time() * 1e-6 * fold
        loop_logger.info(f"[{iteration:04d}] t={solver.time:0.3f} Mzps={Mzps:.3f}")

    yield "end", None, grab_state()


def run(setup_name, quiet=True, **kwargs):
    """
    Run a simulation with no side-effects, and return the final state.

    This function is intended for use by scripts that run a simulation and
    inspect the output in-memory, or otherwise handle archiving the final
    result themselves. Event monitoring is not supported. If `quiet=True`
    (default) then logging is suppressed.
    """
    if "events" in kwargs:
        raise ValueError("events are not supported")

    driver = DriverArgs(setup_name=setup_name, **kwargs)

    if not quiet:
        init_logging()

    return next(simulate(driver))[2]


def init_logging():
    from logging import StreamHandler, Formatter, getLogger, INFO

    class RunFormatter(Formatter):
        def format(self, record):
            name = record.name.replace("sailfish.", "")

            if name == "loop_message":
                return f"{record.msg}"
            if record.levelno <= 20:
                return f"[{name}] {record.msg}"
            else:
                return f"[{name}:{record.levelname.lower()}] {record.msg}"

    handler = StreamHandler()
    handler.setFormatter(RunFormatter())

    root_logger = getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(INFO)


def main():
    import argparse

    init_logging()

    def keyed_event(item):
        key, val = item.split("=")
        return key, Recurrence.from_str(val)

    class MakeDict(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict(values))

    def add_dict_entry(key):
        class AddDictEntry(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                getattr(namespace, self.dest)[key] = values

        return AddDictEntry

    parser = argparse.ArgumentParser(
        prog="sailfish",
        description="gpu-accelerated astrophysical gasdynamics code",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="setup name or restart file",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="print a description of the setup and exit",
    )
    parser.add_argument(
        "--resolution",
        "-n",
        metavar="N",
        type=int,
        help="grid resolution",
    )
    parser.add_argument(
        "--cfl",
        dest="cfl_number",
        metavar="C",
        type=float,
        help="CFL parameter",
    )
    parser.add_argument(
        "--fold",
        "-f",
        metavar="F",
        type=int,
        help="iterations between messages and side effects",
    )
    parser.add_argument(
        "--events",
        nargs="*",
        metavar="E1 E2",
        type=keyed_event,
        action=MakeDict,
        default=dict(),
        help="a sequence of events and recurrence rules to be emitted",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        metavar="C",
        type=Recurrence.from_str,
        action=add_dict_entry("checkpoint"),
        dest="events",
        help="checkpoint recurrence [<delta>|<log:mul>]",
    )
    parser.add_argument(
        "--timeseries",
        "-t",
        metavar="T",
        type=Recurrence.from_str,
        action=add_dict_entry("timeseries"),
        dest="events",
        help="timeseries recurrence [<delta>|<log:mul>]",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        metavar="D",
        type=str,
        dest="output_directory",
        help="directory where checkpoints are written",
    )
    parser.add_argument(
        "--end-time",
        "-e",
        metavar="E",
        type=float,
        help="when to end the simulation",
    )
    exec_group = parser.add_mutually_exclusive_group()
    exec_group.add_argument(
        "--mode",
        dest="execution_mode",
        choices=["cpu", "omp", "gpu"],
        help="execution mode",
    )
    exec_group.add_argument(
        "--use-omp",
        "-p",
        dest="execution_mode",
        action="store_const",
        const="omp",
        help="multi-core with OpenMP",
    )
    exec_group.add_argument(
        "--use-gpu",
        "-g",
        dest="execution_mode",
        action="store_const",
        const="gpu",
        help="gpu acceleration",
    )

    try:
        args = parser.parse_args()

        if args.describe and args.command is not None:
            setup_name = args.command.split(":")[0]
            Setup.find_setup_class(setup_name).describe_class()

        elif args.command is None:
            print("specify setup:")
            for setup in Setup.__subclasses__():
                print(f"    {setup.dash_case_class_name}")

        else:
            driver = DriverArgs.from_namespace(args)

            for name, number, state in simulate(driver):
                if name == "checkpoint":
                    write_checkpoint(number, args.output_directory, state)
                elif name == "end":
                    write_checkpoint("final", args.output_directory, state)
                else:
                    logger.warning(f"unrecognized event {name}")

    except ConfigurationError as e:
        print(f"bad configuration: {e}")

    except SetupError as e:
        print(f"setup error: {e}")

    except ParseRecurrenceError as e:
        print(f"parse error: {e}")

    except OSError as e:
        print(f"file system error: {e}")

    except ModuleNotFoundError as e:
        print(f"unsatisfied dependency: {e}")

    except KeyboardInterrupt:
        print("")
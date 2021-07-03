#define real double

enum ExecutionMode {
    CPU,
    OMP,
    GPU,
};

struct PointMass
{
    real x;
    real y;
    real vx;
    real vy;
    real mass;
    real rate;
    real radius;
};

enum EquationOfStateType
{
    Isothermal,
    LocallyIsothermal,
    GammaLaw,
};

struct EquationOfState
{
    enum EquationOfStateType type;

    union
    {
        struct
        {
            real sound_speed_squared;
        } isothermal;

        struct
        {
            real mach_number_squared;
        } locally_isothermal;

        struct
        {
            real gamma_law_index;
        } gamma_law;
    };
};

enum BufferZoneType
{
    None,
    Keplerian,
};

struct BufferZone
{
    enum BufferZoneType type;

    union
    {
        struct
        {

        } none;

        struct
        {
            real surface_density;
            real central_mass;
            real driving_rate;
            real outer_radius;
            real onset_width;
        } keplerian;
    };
};

struct Mesh
{
    int ni, nj;
    real x0, y0;
    real dx, dy;
};
#define MESH_X(m, i) (m.x0 + (i) * m.dx)
#define MESH_Y(m, i) (m.y0 + (j) * m.dy)
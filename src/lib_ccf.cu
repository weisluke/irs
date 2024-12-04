#include "ccf.cuh"


#if defined(is_float) && !defined(is_double)
using dtype = float; //type to be used throughout this program. float or double
#elif !defined(is_float) && defined(is_double)
using dtype = double; //type to be used throughout this program. float or double
#else
#error "Error. One, and only one, of is_float or is_double must be defined"
#endif

extern "C" 
{
    
    CCF<dtype>* init()                                          {return new CCF<dtype>();}

    void set_kappa_tot(CCF<dtype> *self, dtype val)             {self->kappa_tot                = val;}
    void set_shear(CCF<dtype> *self, dtype val)                 {self->shear                    = val;}
    void set_kappa_star(CCF<dtype> *self, dtype val)            {self->kappa_star               = val;}
    void set_theta_star(CCF<dtype> *self, dtype val)            {self->theta_star               = val;}
    void set_mass_function(CCF<dtype> *self, const char* val)   {self->mass_function_str        = val;}
    void set_m_solar(CCF<dtype> *self, dtype val)               {self->m_solar                  = val;}
    void set_m_lower(CCF<dtype> *self, dtype val)               {self->m_lower                  = val;}
    void set_m_upper(CCF<dtype> *self, dtype val)               {self->m_upper                  = val;}
    void set_rectangular(CCF<dtype> *self, int val)             {self->rectangular              = val;}
    void set_approx(CCF<dtype> *self, int val)                  {self->approx                   = val;}
    void set_safety_scale(CCF<dtype> *self, dtype val)          {self->safety_scale             = val;}
    void set_num_stars(CCF<dtype> *self, int val)               {self->num_stars                = val;}
    void set_starfile(CCF<dtype> *self, const char* val)        {self->starfile                 = val;}
    void set_num_phi(CCF<dtype> *self, int val)                 {self->num_phi                  = val;}
    void set_num_branches(CCF<dtype> *self, int val)            {self->num_branches             = val;}
    void set_random_seed(CCF<dtype> *self, int val)             {self->random_seed              = val;}
    void set_write_stars(CCF<dtype> *self, int val)             {self->write_stars              = val;}
    void set_write_critical_curves(CCF<dtype> *self, int val)   {self->write_critical_curves    = val;}
    void set_write_caustics(CCF<dtype> *self, int val)          {self->write_caustics           = val;}
    void set_write_mu_length_scales(CCF<dtype> *self, int val)  {self->write_mu_length_scales   = val;}
    void set_outfile_prefix(CCF<dtype> *self, const char* val)  {self->outfile_prefix           = val;}

    dtype get_kappa_tot(CCF<dtype> *self)                       {return self->kappa_tot;}
    dtype get_shear(CCF<dtype> *self)                           {return self->shear;}
    dtype get_kappa_star(CCF<dtype> *self)                      {return self->kappa_star;}
    dtype get_theta_star(CCF<dtype> *self)                      {return self->theta_star;}
    const char* get_mass_function(CCF<dtype> *self)             {return (self->mass_function_str).c_str();}
    dtype get_m_solar(CCF<dtype> *self)                         {return self->m_solar;}
    dtype get_m_lower(CCF<dtype> *self)                         {return self->m_lower;}
    dtype get_m_upper(CCF<dtype> *self)                         {return self->m_upper;}
    int get_rectangular(CCF<dtype> *self)                       {return self->rectangular;}
    int get_approx(CCF<dtype> *self)                            {return self->approx;}
    dtype get_safety_scale(CCF<dtype> *self)                    {return self->safety_scale;}
    int get_num_stars(CCF<dtype> *self)                         {return self->num_stars;}
    const char* get_starfile(CCF<dtype> *self)                  {return (self->starfile).c_str();}
    int get_num_phi(CCF<dtype> *self)                           {return self->num_phi;}
    int get_num_branches(CCF<dtype> *self)                      {return self->num_branches;}
    int get_random_seed(CCF<dtype> *self)                       {return self->random_seed;}
    int get_write_stars(CCF<dtype> *self)                       {return self->write_stars;}
    int get_write_critical_curves(CCF<dtype> *self)             {return self->write_critical_curves;}
    int get_write_caustics(CCF<dtype> *self)                    {return self->write_caustics;}
    int get_write_mu_length_scales(CCF<dtype> *self)            {return self->write_mu_length_scales;}
    const char* get_outfile_prefix(CCF<dtype> *self)            {return (self->outfile_prefix).c_str();}

    int get_num_roots(CCF<dtype> *self)                         {return self->get_num_roots();}
    dtype get_corner_x1(CCF<dtype> *self)                       {return self->get_corner().re;}
    dtype get_corner_x2(CCF<dtype> *self)                       {return self->get_corner().im;}
    dtype* get_stars(CCF<dtype> *self)                          {return &(self->get_stars()[0].position.re);}
    dtype* get_critical_curves(CCF<dtype> *self)                {return &(self->get_critical_curves()[0].re);}
    dtype* get_caustics(CCF<dtype> *self)                       {return &(self->get_caustics()[0].re);}
    dtype* get_mu_length_scales(CCF<dtype> *self)               {return self->get_mu_length_scales();}

    bool run(CCF<dtype> *self, int verbose)                     {return self->run(verbose);}
    bool save(CCF<dtype> *self, int verbose)                    {return self->save(verbose);}

}
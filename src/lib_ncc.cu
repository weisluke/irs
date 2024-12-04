#include "ncc.cuh"


#if defined(is_float) && !defined(is_double)
using dtype = float; //type to be used throughout this program. float or double
#elif !defined(is_float) && defined(is_double)
using dtype = double; //type to be used throughout this program. float or double
#else
#error "Error. One, and only one, of is_float or is_double must be defined"
#endif

extern "C" 
{
    
    NCC<dtype>* init()                                          {return new NCC<dtype>();}

    void set_infile_prefix(NCC<dtype> *self, const char* val)   {self->infile_prefix        = val;}
    void set_center_y1(NCC<dtype> *self, dtype val)             {self->center_y.re          = val;}
    void set_center_y2(NCC<dtype> *self, dtype val)             {self->center_y.im          = val;}
    void set_half_length_y1(NCC<dtype> *self, dtype val)        {self->half_length_y.re     = val;}
    void set_half_length_y2(NCC<dtype> *self, dtype val)        {self->half_length_y.im     = val;}
    void set_num_pixels_y1(NCC<dtype> *self, int val)           {self->num_pixels_y.re      = val;}
    void set_num_pixels_y2(NCC<dtype> *self, int val)           {self->num_pixels_y.im      = val;}
    void set_over_sample(NCC<dtype> *self, int val)             {self->over_sample          = val;}
    void set_write_maps(NCC<dtype> *self, int val)              {self->write_maps           = val;}
    void set_write_histograms(NCC<dtype> *self, int val)        {self->write_histograms     = val;}
    void set_outfile_prefix(NCC<dtype> *self, const char* val)  {self->outfile_prefix       = val;}

    const char* get_infile_prefix(NCC<dtype> *self)             {return (self->infile_prefix).c_str();}
    dtype get_center_y1(NCC<dtype> *self)                       {return self->center_y.re;}
    dtype get_center_y2(NCC<dtype> *self)                       {return self->center_y.im;}
    dtype get_half_length_y1(NCC<dtype> *self)                  {return self->half_length_y.re;}
    dtype get_half_length_y2(NCC<dtype> *self)                  {return self->half_length_y.im;}
    int get_num_pixels_y1(NCC<dtype> *self)                     {return self->num_pixels_y.re;}
    int get_num_pixels_y2(NCC<dtype> *self)                     {return self->num_pixels_y.im;}
    int get_over_sample(NCC<dtype> *self)                       {return self->over_sample;}
    int get_write_maps(NCC<dtype> *self)                        {return self->write_maps;}
    int get_write_histograms(NCC<dtype> *self)                  {return self->write_histograms;}
    const char* get_outfile_prefix(NCC<dtype> *self)            {return (self->outfile_prefix).c_str();}

    int* get_num_crossings(NCC<dtype> *self)                    {return self->get_num_crossings();}
    
    bool run(NCC<dtype> *self, int verbose)                     {return self->run(verbose);}
    bool save(NCC<dtype> *self, int verbose)                    {return self->save(verbose);}

}
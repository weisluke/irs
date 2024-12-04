import os
import ctypes


lib = ctypes.CDLL(f'{os.path.dirname(os.path.abspath(__file__))}/../../bin/lib_ncc.so')

lib.init.argtypes = []
lib.init.restype = ctypes.c_void_p

lib.set_infile_prefix.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.set_infile_prefix.restype = ctypes.c_void_p
lib.set_center_y1.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_center_y1.restype = ctypes.c_void_p
lib.set_center_y2.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_center_y2.restype = ctypes.c_void_p
lib.set_half_length_y1.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_half_length_y1.restype = ctypes.c_void_p
lib.set_half_length_y2.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_half_length_y2.restype = ctypes.c_void_p
lib.set_num_pixels_y1.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_num_pixels_y1.restype = ctypes.c_void_p
lib.set_num_pixels_y2.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_num_pixels_y2.restype = ctypes.c_void_p
lib.set_over_sample.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_over_sample.restype = ctypes.c_void_p
lib.set_write_maps.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_maps.restype = ctypes.c_void_p
lib.set_write_histograms.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_histograms.restype = ctypes.c_void_p
lib.set_outfile_prefix.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.set_outfile_prefix.restype = ctypes.c_void_p

lib.get_infile_prefix.argtypes = [ctypes.c_void_p]
lib.get_infile_prefix.restype = ctypes.c_char_p
lib.get_center_y1.argtypes = [ctypes.c_void_p]
lib.get_center_y1.restype = ctypes.c_double
lib.get_center_y2.argtypes = [ctypes.c_void_p]
lib.get_center_y2.restype = ctypes.c_double
lib.get_half_length_y1.argtypes = [ctypes.c_void_p]
lib.get_half_length_y1.restype = ctypes.c_double
lib.get_half_length_y2.argtypes = [ctypes.c_void_p]
lib.get_half_length_y2.restype = ctypes.c_double
lib.get_num_pixels_y1.argtypes = [ctypes.c_void_p]
lib.get_num_pixels_y1.restype = ctypes.c_int
lib.get_num_pixels_y2.argtypes = [ctypes.c_void_p]
lib.get_num_pixels_y2.restype = ctypes.c_int
lib.get_over_sample.argtypes = [ctypes.c_void_p]
lib.get_over_sample.restype = ctypes.c_int
lib.get_write_maps.argtypes = [ctypes.c_void_p]
lib.get_write_maps.restype = ctypes.c_int
lib.get_write_histograms.argtypes = [ctypes.c_void_p]
lib.get_write_histograms.restype = ctypes.c_int
lib.get_outfile_prefix.argtypes = [ctypes.c_void_p]
lib.get_outfile_prefix.restype = ctypes.c_char_p

lib.get_num_crossings.argtypes = [ctypes.c_void_p]
lib.get_num_crossings.restype = ctypes.POINTER(ctypes.c_int)

lib.run.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.run.restype = ctypes.c_bool
lib.save.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.save.restype = ctypes.c_bool

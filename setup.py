from setuptools import setup, find_packages                                                 
from torch.utils.cpp_extension import BuildExtension, CUDAExtension                         
                                                                                            
setup(name='pair_wise_distance_cuda',                                                       
      package_data={'': ['*.so']},                                                          
      include_package_data=True,                                                            
      ext_modules=[                                                                         
          CUDAExtension('pair_wise_distance_cuda', [                                        
              # -- search --                                                                
              'models/pair_wise_distance_cuda_source.cu',                                   
          ],                                                                                
           extra_compile_args={'cxx': ['-g','-w'],                                          
                               'nvcc': ['-O2','-w']})                                       
      ],                                                                                    
      cmdclass={'build_ext': BuildExtension},                                               
)           


cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(CUVReconstruction)

find_package(VTK COMPONENTS 
  vtkCommonColor
  vtkCommonCore
  vtkCommonDataModel
  vtkFiltersCore
  vtkFiltersGeometry
  vtkFiltersModeling
  vtkIOImage
  vtkIOLegacy
  vtkInteractionStyle
  vtkRenderingCore
  vtkRenderingFreeType
  vtkRenderingOpenGL2 QUIET)
if (NOT VTK_FOUND)
  message("Skipping CUVReconstruction: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
if (VTK_VERSION VERSION_LESS "8.90.0")
  # old system
  include(${VTK_USE_FILE})
  add_executable(CUVReconstruction MACOSX_BUNDLE CUVReconstruction.cxx )
  target_link_libraries(CUVReconstruction PRIVATE ${VTK_LIBRARIES})
else ()
  # include all components
  add_executable(CUVReconstruction MACOSX_BUNDLE CUVReconstruction.cxx )
  target_link_libraries(CUVReconstruction PRIVATE ${VTK_LIBRARIES})
  # vtk_module_autoinit is needed
  vtk_module_autoinit(
    TARGETS CUVReconstruction
    MODULES ${VTK_LIBRARIES}
    )
endif () 

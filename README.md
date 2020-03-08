# HeadCT-Reconstruction
This example shows the 3D cardiac ultrasound reconstruction using various hole-filling methods of pixel nearest-neighbour (PNN).

Firstly, the program reads a volume dataset, and removes a whole slice at position n.
User can also choose to remove 2 or 3 slices. The slice removal can be removed at a specific sparsity s. 
Then, the hole-filling method is employed to reconstruct the missing region.

## Pre-requisite
1) Microsoft Visual Studio
2) The Visualization Toolkit (VTK)
3) CMake

## How to run this example?
1) Using CMake to create this project. The procedure is the same as any VTK projects.
2) Build the project using Visual Studio. A new "Debug" folder will be created.
3) Put the volume dataset file (headsq) in the "Debug" folder.
4) Open command prompt and navigate to the "Debug" folder.
5) Run `CUVReconstruction *outputDatasetName(string) *sliceNo(int) *method(string) *parameter(int, float)`
   
   e.g.:
```
CUVReconstruction headsq 7 mean 3 
CUVReconstruction headsq 7 butterfly-my 0 3 0
```
6) Then, input the number of continuous slice to remove.
7) Input the sparsity value.
8) Lastly, input the increment limit of sparsity value. If you want to remove 2 slices in every 7 slices spacing for 10 times: 2 -> 7 -> 10.
9) The result is displayed. The output axial slices are stored in the "figures" folder.

**NOTE:**
1) The dataset used in this example consists of 8-bits grey-scale pixel.
2) The input dataset is preset to "v1" folder. See main().

*For more information, please visit [this conference proceeding](https://doi.org/10.119/GAME47560.2019.8980511) or its [ResearchGate](https://www.researchgate.net/publication/339096910_Using_Modified_Butterfly_Interpolation_Scheme_for_Hole-filling_in_3D_Data_Reconstruction) counterpart.*

The ultrasound dataset used in this project is provided by [Cardia Atlas Project](http://www.cardiacatlas.org/challenges/motion-tracking-challenge/).
For more information about the dataset, please visit [Benchmarking framework for myocardial tracking and deformation algorithms: An open access database](https://www.sciencedirect.com/science/article/pii/S1361841513000388?via%3Dihub).

# Malaria_Screening

This study presents a Frugal framework for automated malaria screening, designed specifically for low-resource hardware. We compared a classical computer vision pipeline against a lightweight Deep Learning benchmark (ShuffleNet). The classical approach evaluated Marker-controlled Watershed and Circle Hough Transform for segmentation. For the classification stage, we trained different classical ML models using optimized statistical color and texture features. Our analysis revealed that while Watershed excels at shape extraction, the Hough Transform provides a more robust total cell count, which is critical for accurate diagnosis. Consequently, the Hough-SVM pipeline achieved a Mean Parasitemia Error of 2.4%, proving its clinical effectiveness despite segmentation noise. Crucially, this classical system required significantly less memory and processing time than deep learning architectures, validating its suitability for deployment in field laboratories where computational power is limited..

## Data
We used dataset from https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html
Named "NLM-Falciparum&Uninfected-Thin-193Patients"

* This dataset includes thin blood smear microscopy images from 193 malaria patients, with 5
images per patient. The data was acquired at Chittagong Medical College Hospital in Bangladesh,
where Giemsa-stained, thin-blood smear images were photographed from P. falciparum-infected
patients and healthy controls using a smartphone camera. The blood smear images were manually
annotated by an expert, de-identified, and archived. 

* The dataset is divided into two parts: a polygon set and a point set. The difference between these
two sets lies in the annotation method. In the polygon set, all red blood cells (RBCs) and white
blood cells (WBCs) have been outlined manually with polygons using the Firefly annotation tool,
whereas in the point set, cells have been marked by placing a point on each cell.

### Masks Generation
We used *Polygon Set* to prepare the Mask Set, where we converted the cell boundary coordinates from text files into binary masks, where the cells are filled white and the background is black. To preserve the cell boundary information, we outlined each cell with a black border.

### Full Pipeline

1. Preprocessing (Green channel + CLAHE)

3. Segmentation
   - Otsu
   - Watershed
   - Hough Transform
      
4. Cell Cropping
      
5. Feature Extraction
      
6. ML Classification (LR, RF, SVM)
      
7. Parasitemia Estimation

#### Find the detailed report in Report_Soloveva_Marino.pdf

### Final Data Folder structure
```text
NIH-NLM-ThinBloodSmearsPf/
├── Point Set/                  # Dataset with point annotations (used for counting/classification)
│   ├── 143C39P4thinF/
│   └── ...
│
└── Polygon Set/                # Dataset with polygon annotations (used for segmentation)
    ├── 150C49P10thinF/         # Folder for a specific Patient ID
    │   ├── GT/                 # Ground Truth (Raw Annotations)
    │   │   ├── IMG_20150724_102330.txt
    │   │   └── ...
    │   │
    │   ├── Img/                # Original Microscope Images (JPEG, RGB)
    │   │   ├── IMG_20150724_102330.jpg
    │   │   └── ...
    │   │
    │   └── Masks/              # GENERATED Binary Masks (PNG, Black & White)
    │       ├── IMG_20150724_102330.png
    │       └── ...
    │
    └── ...

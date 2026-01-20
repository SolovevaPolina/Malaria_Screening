# Malaria_Screening

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
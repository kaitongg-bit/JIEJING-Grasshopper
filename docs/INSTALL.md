# Installation

## Rhino 8 / Grasshopper

1. Download and unzip the latest JIEJING release.
2. In Grasshopper, open **File → Special Folders → Components Folder**.
3. Copy `JIEJING.Components.gha` into that folder.
4. Optional: copy the four files in `assets/selectors/` into Grasshopper's
   UserObjects folder.
5. Restart Rhino and Grasshopper.
6. Confirm that the red **JIEJING** tab appears.

On Windows, if Grasshopper blocks the downloaded GHA, open the file's Properties and
select **Unblock**, then restart Rhino.

## Beginner workflow

Open `starter/JIEJING_Baby_Starter.gh`. The file is already wired and can generate a
traditional lattice without an image.

To add an image:

1. Right-click the **IMAGE FILE** component.
2. Choose **Select one existing file**.
3. Select a JPG or PNG.
4. Choose Silhouette, Portrait Detail, or Hybrid.
5. Adjust Image Strength and Material/Opening polarity.

The Export toggle is deliberately `False` on startup. Inspect the black-and-white
material preview first, then switch Export to `True`. Output is written to
`~/Documents/JIEJING_exports/`.


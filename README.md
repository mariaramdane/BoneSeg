- The detailed procedure for data acquisition is in "procedure_acquisition.pdf".
- To test the app, download the sample of data here: https://doi.org/10.5281/zenodo.18236099
- Download all the files in the main folder of the repository, and put them in the same folder containing your data and sam2 folders. Check "DMP_document.pdf" for more details on folders architecture.
- Click on the files in this order :
1) setup_me.bat
2) build_app.bat
The application will be automatically created after this. Click on the new file BoneSeg.exe to launch the interface.
- Other codes used for the project have been provided in this repository:
* run_pca.bat, pca.py and results.txt to perform Principal Component Analysis on the results
* macro.ijm for ImageJ analysis (histogram plot of BrightField images)
* QuPath_annotation_to_png_mask.groovy for binary mask creation used in SAM2 training.

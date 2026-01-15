- The detailed procedure for data acquisition is in "procedure_acquisition.pdf".
- To test the app, download the sample of data here: https://doi.org/10.5281/zenodo.18236099
- Please first check DMP_document.pdf for details on folders architecture.
- Download all the files of this repository, and put them in the same folder containing your data and sam2 folders. 
- Then, click on the files in this order :
1) setup_me.bat
2) build_app.bat
- The application will be automatically created after this. Click on the new file BoneSeg.exe to launch the interface.
- Other codes used for the project have been provided in this repository:
1) run_pca.bat, pca.py and results.txt to perform Principal Component Analysis on the results
2) macro.ijm for ImageJ analysis (histogram plot of BrightField images)
3) QuPath_annotation_to_png_mask.groovy for binary mask creation used in SAM2 training.
4) icon.ico and logo.png are for the display of the logo on the interface.

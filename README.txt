Welcome to Accel-Wrapper

The script accel_pipe.py is designed to convert your csv or xlsx files of raw accelerometry data
into the "XYZ,XYZ,XYZ..." format that can be dropped into the AcceleRater website: http://accapp.move-ecol-minerva.huji.ac.il/
to be used to train various Machine Learning algorithms on your data. 

Creating the ML training/testing dataset:
Input csv/xlsx requirements:
	 4 columns labeled: Behavior, accX, accY, accZ
	rows with missing values in any of the four columns will be ignored. 

If using the -r (wild-data) flag the file must be in .xlsx format

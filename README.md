# Frequency Analysis

This code computes the psi statistic periodogram, defined as two times the lomb-scargle power over the Lafler–Kinman statistic.
The following command is an example of how to run the code with MeerLICHT data:

In the above example, the code computes the psi statistic periodogram of each object in the file unique_ra_dec.csv. The lightcurves of each object in this file are stacked in the lightcurve.csv file, which contains the same RA and DEC information as the unique_ra_dec.csv file. Both files should be in the same directory (defined by the argument --directory). The corresponding periodograms are saved on the path defined by the --save_to_path argument. Other parameters, such as the Nyquist frequency and filters, are explained using the following command: python psi_stat.py -h

For more information about the method used in this code, please visit 

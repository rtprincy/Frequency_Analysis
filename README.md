# Frequency analysis

This code computes the psi statistic periodogram, defined as two times the lomb-scargle power over the Lafler–Kinman statistic.
The following command is an example of how to run the code with MeerLICHT data:

python psi_stat.py --name_data_file "unique_ra_dec.csv" --name_lc_file "lightcurve.csv" --directory "directory/" --save_to_path "periodograms/" --maximum_frequency=480 

In the above example, the code computes the psi statistic periodogram of each object in the file unique_ra_dec.csv. The lightcurves of each object in this file are stacked in the lightcurve.csv file, which contains the same RA and DEC information as the unique_ra_dec.csv file. Both files should be in the same directory (defined by the argument --directory), and either a csv or fits file is accepted. The corresponding periodograms are saved on the path defined by the --save_to_path argument. Other parameters, such as the Nyquist frequency and filters, are explained using the following command: python psi_stat.py -h

The method used in this code is described by [Ranaivomanana et al., 2023](https://doi.org/10.1051/0004-6361/202245560)

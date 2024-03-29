# Frequency analysis

This code computes the psi statistic periodogram, defined as two times the lomb-scargle power over the Lafler–Kinman statistic.
The following command is an example of how to run the code with BlackGEM data:

python psi_statistic.py --source_id "gaia_id.csv" --name_lc_file "bg_lightcurves.csv" --directory "/users/princy/" --save_to_path "/users/princy/periodogram/" --minimum_frequency=0.05 --maximum_frequency=50 --passbands='q' 

In the above example, the code computes the psi statistic periodogram of each object in the source id file "gaia_id.csv". It can also take a single Gaia id value. The lightcurves of each object in this file are stacked in the bg_lightcurves.csv file, which contains the same Gaia id as in the source id file. Both files should be in the same directory (defined by the argument --directory), and either a csv or fits file is accepted. The corresponding periodograms are saved to the path defined by the --save_to_path argument. Other parameters, such as the Nyquist frequency and filters, are explained using the following command: python psi_stat_bgdata.py -h

The method used in this code is described by [Ranaivomanana et al., 2023](https://doi.org/10.1051/0004-6361/202245560)


For a single Gaia id, replace "gaia_id" with its actual value:


python psi_stat_bgdata.py --source_id gaia_id --name_lc_file "bg_lightcurves.csv" --directory "/users/princy/" --save_to_path "/users/princy/periodogram/" --minimum_frequency=0.05 --maximum_frequency=50 --passbands='q,u,i' 

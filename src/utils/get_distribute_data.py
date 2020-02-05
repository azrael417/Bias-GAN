import os
import cdsapi

#open client
c = cdsapi.Client()

#parameters
dataset_name = 'reanalysis-era5-single-levels'
variables = [ '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
              '2m_temperature', 'mean_sea_level_pressure', 'mean_wave_direction',
              'mean_wave_period', 'sea_surface_temperature', 'significant_height_of_combined_wind_waves_and_swell',
              'surface_pressure', 'total_precipitation']
months = ["{:02d}".format(x) for x in range(1,13)]
days = ["{:02d}".format(x) for x in range(1,32)]
times = ["{:02d}:00".format(x) for x in range(0,24)]
years = ['1979', '1980', '1981', '1982']

#loop over years
for idx, year in enumerate(years):
    for idy, month in enumerate(months):
        
        #distribute round robin in years
        path = '/data{}/ecmwf_data/gpu{}'.format(idx, (idy // 3) + idx * 4)
        
        #filename
        filename = 'download_'+year+'-'+month+'.nc'

        #outputfile
        outputfile = os.path.join(path,filename)

        #create folder
        if not os.path.isdir(path):
            os.makedirs(path)
        
        #get data
        c.retrieve(
            dataset_name,
            {
                'product_type': 'reanalysis',
                'variable': variables,
                'year': [year],
                'month': [month],
                'day': days,
                'time': times,
                'format': 'netcdf'
            },
            outputfile)
        break


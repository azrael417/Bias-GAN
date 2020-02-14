import os
import cdsapi

#some raid mod stuff
nraid = 4

#open client
c = cdsapi.Client()

#parameters
root_dir = "/"
dataset_name = 'reanalysis-era5-single-levels'
variables = [ '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
              '2m_temperature', 'mean_sea_level_pressure', 'mean_wave_direction',
              'mean_wave_period', 'sea_surface_temperature', 'significant_height_of_combined_wind_waves_and_swell',
              'surface_pressure', 'total_precipitation']
times = ["{:02d}:00".format(x) for x in range(0,24)]
days = ["{:02d}".format(x) for x in range(1,32)]
months = ["{:02d}".format(x) for x in range(1,13)]
#years = ['1979', '1980', '1981', '1982']
years = ['1983', '1984', '1985', '1986']

#loop over years
for idx, year in enumerate(years):
    for idy, month in enumerate(months):

        #distribute round robin in years 
        path = os.path.join(root_dir,'data{}'.format(2 * (idx % nraid) + 1), 'ecmwf_data', 'gpu{}'.format((idy // 3) + idx * 4))

        print(path)
        exit
        
        #create folder
        if not os.path.isdir(path):
            os.makedirs(path)

        #loop over days
        for idz, day in enumerate(days):
        
            #filename
            filename = dataset_name+'_'+year+'-'+month+'-'+day+'.nc'
            
            #outputfile
            outputfile = os.path.join(path,filename)

            #skip if file exists
            if os.path.isfile(outputfile):
                print("file {} already exist".format(outputfile))
                continue
            
            #get data
            try:
                c.retrieve(
                    dataset_name,
                    {
                        'product_type': 'reanalysis',
                        'variable': variables,
                        'year': [year],
                        'month': [month],
                        'day': [day],
                        'time': times,
                        'format': 'netcdf'
                    },
                    outputfile)
            except:
                print("Unable to retrieve information for {}-{}-{}".format(year, month, day))
                continue

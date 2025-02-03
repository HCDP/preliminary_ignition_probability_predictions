import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import rasterio as rio
from matplotlib.colors import ListedColormap
import os
import sys
import pytz
from datetime import datetime,timedelta

NO_DATA_VAL = float(os.environ.get('NO_DATA_VAL'))

def initialize_directories(county):
    master_dir = os.environ.get('PROJECT_ROOT')
    dep_dir = os.environ.get('DEPENDENCY_DIR')
    init_data_path = dep_dir + "FireData-2002-2019_"+county.lower()+".csv"
    init_geo_ref = dep_dir + "ref_"+county.lower()+".tif"
    init_lc_path = dep_dir + "PerCov2016model_"+county.lower()+".tif"
    
    init_rh_path = dep_dir + "RH_"+county.lower()+".tif"
    init_tmax_path = dep_dir + "Tmax_"+county.lower()+".tif"
    init_ndvi_path = dep_dir + "NDVI_"+county.lower()+".tif"
    init_pre_path = dep_dir + "Preciptation_"+county.lower()+".tif"
    init_api_path = dep_dir + "API_"+county.lower()+".tif"
    init_clim_path = dep_dir
    init_out_path = master_dir + "data_outputs/day/"

    model = WildfireRiskModel(
        data_path=init_data_path, #dependency bundle
        ref_geotiff_path=init_geo_ref, #dep bundle
        rh_path=init_rh_path, #wget from prior workflow
        lc_raster_path=init_lc_path, #dep bundle
        tmax_path = init_tmax_path, #wget
        ndvi_path=init_ndvi_path, #wget
        pre_path= init_pre_path, #wget (precip)
        api_path=init_api_path, #computed local within workflow (action needed)
        climat_data_folder=init_clim_path, #computed local within workflow (already done)
        output_folder=init_out_path, #data_output location for ingestion
    )
    return model

class WildfireRiskModel:
    def __init__(self, data_path, ref_geotiff_path, rh_path, lc_raster_path, tmax_path, ndvi_path, pre_path, api_path, climat_data_folder, output_folder):
        ''' Initialize the model with paths for data, GeoTIFF, climate data, and output folder'''
        self.data_path = data_path
        self.ref_geotiff_path = ref_geotiff_path
        self.climat_data_path = climat_data_folder
        self.output_folder = output_folder
        
        self.rh_path = rh_path
        self.lc_raster_path = lc_raster_path
        self.tmax_path = tmax_path
        self.ndvi_path = ndvi_path
        self.pre_path = pre_path
        self.api_path = api_path
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.RF = None

        # Initialize coordinates based on GeoTIFF
        self.x_coordinates, self.y_coordinates = self.generate_coordinates_from_geotiff()

    def generate_coordinates_from_geotiff(self):
        ''' Generate x and y coordinate matrices based on the reference GeoTIFF'''
        with rio.open(self.ref_geotiff_path) as ref_geotiff:
            x_coordinates = np.zeros((ref_geotiff.height, ref_geotiff.width)) + np.nan
            y_coordinates = np.zeros((ref_geotiff.height, ref_geotiff.width)) + np.nan
            for i in range(ref_geotiff.height):
                for j in range(ref_geotiff.width):
                    x_coordinates[i, j], y_coordinates[i, j] = ref_geotiff.xy(i, j)
        return x_coordinates, y_coordinates


    def load_and_preprocess_data(self):
        ''' Load the main dataset, clean missing values, and split into train and test sets '''
        data = pd.read_csv(self.data_path)
        self.df = data.replace([-32768, -9999], np.nan).dropna(ignore_index=True)
        X = self.df.iloc[:, :-1] # Load and preprocess data, ensuring the last column is the target class (fire vs. non-fire), where values are 0 (non-fire) or 1 (fire)
        Y = self.df.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.3, random_state=0)     

    def train_model(self):
        ''' Train a Random Forest classifier on the training data '''
        self.RF = RandomForestClassifier(random_state=1)
        self.RF.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        '''Evaluate the trained model and print performance metrics '''
        probRF = self.RF.predict_proba(self.X_test)[:, 1]
        sensitivity, specificity, npv, ppv, auc = self.calculate_metrics(self.y_test, probRF)
        print(f"Sensitivity: {sensitivity}\nSpecificity: {specificity}\nNPV: {npv}\nPPV: {ppv}\nAUC: {auc}")

    def calculate_metrics(self, y_true, y_pred_probs, threshold=0.5):
        ''' Calculate sensitivity, specificity, NPV, PPV, and AUC metrics for model evaluation '''
        y_pred = (y_pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn)
        ppv = tp / (tp + fp)
        auc = roc_auc_score(y_true, y_pred_probs)
        return sensitivity, specificity, npv, ppv, auc
    
    def extract_climate_data(self, targ_file, target_date, var_name, nodata_value):
        '''
        Extract climate data for a given variable and date 
        NRT note: set filename to data-agnostic so that date manip not required
        [target_date] deprecate
        Targ_file is part of object instantiation
        '''
        target_data = []
        print(f"Processing {targ_file}")
        with rio.open(targ_file, "r") as raster:
            target_data = raster.read(1).flatten()

        df = pd.DataFrame()
        df[var_name] = target_data
        df[var_name] = df[var_name].replace(nodata_value, np.nan)

        return df
 
    
    def extract_lcband_data(self):
        ''' Extract land cover bands for each pixel '''
        LC_band1, LC_band2, LC_band3 = [], [], []

        df = pd.DataFrame()

        with rio.open(self.lc_raster_path) as lc_raster:
            
            band1 = lc_raster.read(1)
            band2 = lc_raster.read(2)
            band3 = lc_raster.read(3)
            for i in range(self.x_coordinates.shape[0]):
                for j in range(self.x_coordinates.shape[1]):
                    m, n = lc_raster.index(self.x_coordinates[i, j], self.y_coordinates[i, j])

                    LC_band1.append(band1[m, n])
                    LC_band2.append(band2[m, n])
                    LC_band3.append(band3[m, n])

        df['LC_band1'] = np.array(LC_band1)
        df['LC_band2'] = np.array(LC_band2)
        df['LC_band3'] = np.array(LC_band3)
        df = df.replace([-32768], np.nan)

        return df

    def process_all_dates(self, current_date):
        ''' Process and save climate and land cover data for given date'''
        
        date_str = current_date.strftime('%Y_%m_%d')
        year_str = current_date.strftime('%Y')
        print(f"Processing date: {date_str}")
        
        try:
            # Extract data for each variable
            df_rh = self.extract_climate_data(self.rh_path, date_str, 'RH', [-9999])
            df_lc = self.extract_lcband_data()
            df_tmax = self.extract_climate_data(self.tmax_path, date_str, 'Tmax', [-9999])
            df_ndvi = self.extract_climate_data(self.ndvi_path, date_str, 'NDVI', [-3.3999999521443642e+38])
            df_precipitation = self.extract_climate_data(self.pre_path, date_str, 'Precipitation', [-3.3999999521443642e+38])
            df_api = self.extract_climate_data(self.api_path, date_str, 'API', [-9999])
            #set coordinate df
            df_latlon = pd.DataFrame([[x,y] for x,y in zip(self.x_coordinates.flatten(),self.y_coordinates.flatten())],columns=['lon','lat'])
            
            # Merge all dataframes
            df = pd.concat([df_latlon,df_rh, df_lc, df_tmax, df_ndvi, df_precipitation, df_api], axis=1)

            # Save to CSV
            output_path = os.path.join(self.climat_data_path, f"processed_input.csv")
            df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"An error occurred on {date_str}: {e}")


    def compute_probability(self, file_path):
        ''' Load and process external climate data for risk prediction '''
        df1 = pd.read_csv(file_path)
        df2 = df1.dropna()
        fire = df2.drop(columns=['lon', 'lat'], errors='ignore')
        fire.columns = ['RH', 'LC_band1', 'LC_band2', 'LC_band3', 'Tmax', 'NDVI', 'Precipitation', 'API']
        probRF = self.RF.predict_proba(fire)[:, 1]
        prob = pd.DataFrame(probRF, columns=['prob'], index=df2.index)
        prob_final = pd.DataFrame(prob['prob'] - (df1['lat'] / 1e11), columns=['prob'])
        return prob_final

    def create_fire_risk_map(self, prob_final, title, date):
        ''' Create a visual fire risk map using the computed risk probabilities '''
        with rio.open(self.ref_geotiff_path) as ref_geotiff:
            ref_data = ref_geotiff.read(1)
            bbox = ref_geotiff.bounds
            height, width = ref_data.shape
            Prob_2d_clf = prob_final['prob'].values.reshape(ref_data.shape)
            hsv_modified = plt.get_cmap('nipy_spectral', 256)
            newcmp = ListedColormap(hsv_modified(np.linspace(0.5, 0.9, 256)))
            fig, ax = plt.subplots(figsize=(8,6), dpi=400)
            plt.imshow(Prob_2d_clf, extent=bbox, cmap=newcmp, zorder=1)
            cbar = plt.colorbar(fraction=0.047*(height/width), label='Fire risk')
            plt.title(title, size=24)
            plt.xlabel('Longitude', size=18)
            plt.ylabel('Latitude', size=18)
            plt.tight_layout()
    
    def create_fire_risk_maps_for_dates(self, current_date,county):
        ''' Generate fire risk maps for each date within the specified date range'''
        year = current_date.strftime('%Y')
        date_str = current_date.strftime('%Y_%m_%d')
        file_path = f"{self.climat_data_path}/processed_input.csv"

        try:
            if os.path.exists(file_path):
                print('Running compute_probability')
                prob_final = self.compute_probability(file_path)
                png_output_folder = f'{self.output_folder}/png'
                geotiff_dir = f'{self.output_folder}/tiff'
                os.makedirs(png_output_folder, exist_ok=True)
                os.makedirs(geotiff_dir, exist_ok=True)
                self.create_fire_risk_map(prob_final, f'Day {date_str}', date_str)
                plt.savefig(os.path.join(png_output_folder, f'Probability_{county}.png'))
                plt.close()
                geotiff_path = os.path.join(geotiff_dir, f'Probability_{county}.tif')

                #src_path = f'{self.pre_path}//rainfall_new_day_oa_data_map_2019_12_01.tif'
                prob_final[np.isnan(prob_final.values)] = NO_DATA_VAL
                #src_path = self.tmax_path
                src_path = self.ref_geotiff_path
                with rio.open(src_path) as src:
                    ref_geotiff_meta = src.profile
                    ref_geotiff_meta.update(compress='lzw')
                    ref_geotiff_meta.update(nodata=NO_DATA_VAL)
                    with rio.open(geotiff_path, 'w', **ref_geotiff_meta) as dst:
                        dst.write(prob_final.values.reshape(src.height, src.width), 1)
            else:
                print(f"File not found for date {date_str}, skipping...")
        except Exception as e:
            print(f"An error occurred for date {date_str}: {e}")

if __name__=="__main__":
    #explicit extent variable definition required. call from shell or make wrapper.
    if len(sys.argv) > 2:
        extent = sys.argv[1]
        input_date = sys.argv[2]
        process_date = pd.to_datetime(input_date) #converts any format to single datatype
        date_str = process_date.strftime('%Y-%m-%d') #converts date to standardized format
    else:
        extent = sys.argv[1]
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        process_date = today - timedelta(days=1)
        date_str = process_date.strftime('%Y-%m-%d')
    
    model = initialize_directories(extent)
    model.load_and_preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.process_all_dates(process_date)
    model.create_fire_risk_maps_for_dates(process_date,extent)
    

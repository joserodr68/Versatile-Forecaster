
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from statsmodels.tsa.stattools import acf, pacf, adfuller
import pmdarima as pm
import arch  
from neuralforecast import NeuralForecast
from prophet import Prophet
from statsforecast import StatsForecast
from neuralforecast.models import NHITS
from neuralforecast.models import Informer
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss, MAE, RMSE, MSE
from typing import Dict, Any

from aux_langchain import *



# Set the environment variable to address a FutureWarning
os.environ['NIXTLA_ID_AS_COL'] = 'True'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize a context storage in the FastAPI state
app.state.context_info: Dict[str, Any] = {}


# function to manage context updates
def update_context(key: str, data: Any) -> None:
    """
    Update the context information for a specific analysis type
    
    Args:
        key (str): Type of analysis (e.g., 'statistics', 'arima', 'forecast')
        data (Any): The data to store
    """
    app.state.context_info[key] = data


def regularize_timeseries(df, timestamp_col, values_col, is_prices=False):
    """Create a regular daily time series, filling missing dates with zeros unless it is a prices series (stocks etc)"""
    # Set timestamp as index
    df = df.set_index(timestamp_col)
    
    # Create a complete date range
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    
    if is_prices:
        # Reindex and fill missing values with the last value
        regular_df = df.reindex(date_range, method='ffill')
    else:
        # Reindex and fill missing values with 0
        regular_df = df.reindex(date_range, fill_value=0)
    
    # Reset index and rename columns
    regular_df = regular_df.reset_index()
    regular_df.columns = [timestamp_col, values_col]
    
    return regular_df

def fit_auto_arima(values):
    """Fit AutoARIMA model using pmdarima and return parameters and predictions"""
    try:
        # Fit the model
        model = pm.auto_arima(values,
                            start_p=1, start_q=1,
                            test='adf',
                            max_p=10, max_q=10,
                            m=1,
                            d=None,
                            seasonal=False,
                            start_P=0,
                            D=0,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
        
        # Get in-sample predictions
        predictions = model.predict_in_sample()
        
        return {
            "predictions": predictions.tolist(),
            "order": list(model.order),  # (p,d,q)
            "aic": float(model.aic()),
            "info": str(model.summary())
        }
    except Exception as e:
        logger.error(f"Error in AutoARIMA fitting: {str(e)}", exc_info=True)
        return None

def perform_adf_test(values):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    try:
        result = adfuller(values)
        return {
            "test_statistic": float(result[0]),
            "p_value": float(result[1]),
            "critical_values": {str(key): float(value) for key, value in result[4].items()},
            "is_stationary": result[1] < 0.05  # True if p-value < 0.05
        }
    except Exception as e:
        logger.error(f"Error performing ADF test: {str(e)}")
        return None

def calculate_acf_pacf(values, n_lags=40):
    """Calculate ACF and PACF using statsmodels"""
    try:
        n_lags = min(n_lags, len(values) - 1)
        
        # Calculate ACF
        acf_values = acf(values, nlags=n_lags, fft=True)
        
        # Calculate PACF
        pacf_values = pacf(values, nlags=n_lags)
        
        # Calculate confidence intervals (95%)
        conf_int = 1.96 / np.sqrt(len(values))
        
        return {
            "acf": acf_values.tolist(),
            "pacf": pacf_values.tolist(),
            "conf_int": float(conf_int)
        }
    except Exception as e:
        logger.error(f"Error calculating ACF/PACF: {str(e)}")
        return None

def fit_garch(data):
    """Fit GARCH model and return parameters and volatility"""
    try:
        # Scale the data to avoid convergence issues
        scale_factor = 1e-5 if np.mean(np.abs(data)) > 1000 else 1
        scaled_data = data * scale_factor
        
        model = arch.arch_model(scaled_data, vol='Garch', p=1, q=1, rescale=False)
        result = model.fit(disp='off')
        
        # Rescale the volatility back to original scale
        volatility = result.conditional_volatility / scale_factor
        
        return {
            "params": {
                "omega": float(result.params['omega']),
                "alpha[1]": float(result.params['alpha[1]']),
                "beta[1]": float(result.params['beta[1]'])
            },
            "volatility": volatility.tolist()
        }
    except Exception as e:
        logger.error(f"Error in GARCH fitting: {str(e)}")
        return None
        

# ------------------------------------------------------------------------------------------


def split_data(df, timestamp_col, values_col, train_size=0.85):
    """Split the data into training and validation sets"""
    n = len(df)
    train_size = int(n * train_size)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()
    
    return train_df, val_df

def calculate_metrics(actual, predicted):
    """Calculate forecast performance metrics"""
    return {
        'mae': float(mean_absolute_error(actual, predicted)),
        'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        'r2': float(r2_score(actual, predicted)),
        'mape': float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    }



def fit_prophet(train_df, val_df, timestamp_col, values_col, forecast_periods):
    """Fit and predict using Prophet"""
    try:
        logger.info("Starting Prophet model fitting")
        
        # Prepare data for Prophet
        prophet_train = train_df.rename(columns={
            timestamp_col: 'ds',
            values_col: 'y'
        })
        
        # Initialize and fit model
        model = Prophet(yearly_seasonality=True, 
                       weekly_seasonality=True,
                       daily_seasonality=False)
        model.fit(prophet_train)
        
        # First, create future dataframe just for validation
        future_val = model.make_future_dataframe(
            periods=len(val_df),
            freq='D'
        )
        
        # Get validation predictions
        forecast_val = model.predict(future_val)
        val_predictions = forecast_val['yhat'].iloc[-len(val_df):].values
        
        # Now create future dataframe for actual forecast beyond the entire dataset
        future_forecast = model.make_future_dataframe(
            periods=forecast_periods,
            freq='D',
            include_history=False
        )
        future_forecast['ds'] = pd.date_range(
            start=val_df[timestamp_col].max() + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        # Generate future predictions
        forecast_future = model.predict(future_forecast)
        future_predictions = forecast_future['yhat'].values
        future_dates = future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        
        # Calculate metrics on validation set
        metrics = calculate_metrics(val_df[values_col].values, val_predictions)
        
        logger.info("Preparing Prophet results")
        results = {
            'train_dates': train_df[timestamp_col].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_df[values_col].tolist(),
            'val_dates': val_df[timestamp_col].dt.strftime('%Y-%m-%d').tolist(),
            'val_values': val_df[values_col].tolist(),
            'validation_predictions': val_predictions.tolist(),
            'forecast_dates': future_dates,
            'forecast_predictions': future_predictions.tolist(),
            'metrics': metrics
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in Prophet forecasting: {str(e)}")
        logger.exception("Full traceback:")
        return None

def fit_nhits(train_df, val_df, timestamp_col, values_col, forecast_periods):
    """Fit and predict using N-HITS"""
    try:
        # Prepare data for NHITS
        data = train_df.rename(columns={
            timestamp_col: 'ds',
            values_col: 'y'
        }).copy()
        data['unique_id'] = 'series'
        data['ds'] = pd.to_datetime(data['ds'])
        data = data[['ds', 'unique_id', 'y']]

         # Generate future dates for validation + forecast
        future_dates = pd.date_range(
            start=val_df[timestamp_col].iloc[0],  # Start from beginning of validation
            end=val_df[timestamp_col].iloc[-1] + pd.Timedelta(days=forecast_periods),
            freq='D'
        )
        future_df = pd.DataFrame({
            'ds': future_dates,
            'unique_id': 'series'
        })

        # Initialize model just for validation + forecast periods
        model = NeuralForecast(
            models=[NHITS(
                h=len(val_df) + forecast_periods,
                input_size=30,
                max_steps=350
            )],
            freq='D'
        )
        
        # Fit model
        model.fit(data)
        
       
        # Get predictions
        predictions_df = model.predict(futr_df=future_df)
        
        # Split predictions into validation and forecast periods
        val_predictions = predictions_df['NHITS'].iloc[:-forecast_periods].values
        future_predictions = predictions_df['NHITS'].iloc[-forecast_periods:].values
        
        # Get forecast dates (starting after validation period)
        forecast_dates = future_dates[-forecast_periods:].strftime('%Y-%m-%d').tolist()
        
        logger.info(f"Validation data length: {len(val_df)}")
        logger.info(f"Validation predictions length: {len(val_predictions)}")
        logger.info(f"Future predictions length: {len(future_predictions)}")
        logger.info(f"First forecast date: {forecast_dates[0]}")
        logger.info(f"Last forecast date: {forecast_dates[-1]}")
        
        # Calculate metrics on validation set
        metrics = calculate_metrics(
            val_df[values_col].values,
            val_predictions
        )
        
        return {
            'validation_predictions': val_predictions.tolist(),
            'forecast_predictions': future_predictions.tolist(),
            'forecast_dates': forecast_dates,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Error in NHITS forecasting: {str(e)}")
        logger.exception("Full traceback:")
        return None


def fit_informer(train_df, val_df, timestamp_col, values_col, forecast_periods):
    """Fit and predict using Informer"""
    try:
        # Prepare data for Informer
        
        data = train_df.rename(columns={
            timestamp_col: 'ds',
            values_col: 'y'
        }).copy()
        data['unique_id'] = 'series'
        data['ds'] = pd.to_datetime(data['ds'])
        data = data[['ds', 'unique_id', 'y']]


        # Generate future dates for validation + forecast
        future_dates = pd.date_range(
            start=val_df[timestamp_col].iloc[0],  # Start from beginning of validation
            end=val_df[timestamp_col].iloc[-1] + pd.Timedelta(days=forecast_periods),
            freq='D'
        )
        future_df = pd.DataFrame({
            'ds': future_dates,
            'unique_id': 'series'
        })

        model = Informer(h=len(val_df) + forecast_periods,          
                 input_size = forecast_periods,
                 hidden_size = 512,   # int=128, units of embeddings and encoders.
                 conv_hidden_size = 128,  #int=32, channels of the convolutional encoder.
                 n_head = 128,
                 loss=MAE(),
                 scaler_type='robust',
                 learning_rate=1e-3,
                 max_steps=50,
                 encoder_layers = 2, # Número de capas para el decodificador TCN. int = 2
                 decoder_layers = 2,
        )

        nf = NeuralForecast(
            models=[model],
            freq='D'
        )

        # Fit model
        nf.fit(df=data)

        

        predictions = nf.predict(futr_df=future_df)

        # # Split predictions into validation and forecast
        val_predictions = predictions.iloc[:-forecast_periods]
        future_predictions = predictions.iloc[-forecast_periods:]
        
        # Calculate metrics
        metrics = calculate_metrics(val_df.iloc[:, 1].values, val_predictions['Informer'].values)


        return {
          'validation_predictions': val_predictions['Informer'].tolist(),
          'forecast_predictions': future_predictions['Informer'].tolist(),
          'forecast_dates': future_dates[-forecast_periods:].strftime('%Y-%m-%d').tolist(),
          'metrics': metrics
        }
        

        return predictions
    except Exception as e:
        print(f"Error in Informer forecasting: {str(e)}")
        return None



@app.post("/generate-report/{filename}")
async def generate_report(filename: str, request: Request):
    try:
        # Get context info from app state
        context_info = request.app.state.context_info
        
        # Generate the report
        report = generate_time_series_report(context_info)
        
        print(report)

        return JSONResponse({
            "status": "success",
            "report": report
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Received file upload request: {file.filename}")
    
    if not file.filename.endswith('.csv'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Save the file
        file_path = UPLOAD_DIR / file.filename
        logger.debug(f"Saving file to: {file_path}")
        
        content = await file.read()
        file_path.write_bytes(content)
        
        # Read and validate the CSV
        logger.debug("Reading CSV file for validation")
        df = pd.read_csv(file_path)
        logger.debug(f"CSV columns: {df.columns.tolist()}")

        app.state.context_info = {}
        
        return {"filename": file.filename, "status": "File uploaded successfully"}
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/analyze/{filename}")
async def analyze_file(filename: str):
    logger.info(f"Starting analysis for file: {filename}")
    try:
        # Read the CSV file
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            logger.warning(f"File not found: {filename}")
            raise HTTPException(status_code=404, detail="File not found")
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        logger.debug(f"CSV loaded successfully. Columns: {df.columns.tolist()}")
        
        # Try to identify timestamp and values columns
        timestamp_col = next((col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'price' in col.lower()), None)  # ojo
        values_col = next((col for col in df.columns if 'value' in col.lower() or 'sales' in col.lower() or 'price' in col.lower()), None)

        if any(col.lower() == 'price' for col in df.columns):
            is_prices = True
            update_context('series type', 'prices sequence')
        else:
            is_prices = False
            update_context('series type', 'values sequence')

        if not timestamp_col or not values_col:
            logger.warning(f"Could not identify timestamp or values columns. Using first two columns.")
            timestamp_col = df.columns[0]
            values_col = df.columns[1]
            
        # Convert timestamp
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Convert values to numeric and handle missing values
        df[values_col] = pd.to_numeric(df[values_col], errors='coerce')
        df = df.dropna()
        
        # Sort by timestamp and regularize the series
        df = df.sort_values(timestamp_col)
        df = regularize_timeseries(df, timestamp_col, values_col, is_prices)
        
        # Calculate basic statistics
        stats = {
            "mean": float(df[values_col].mean()),
            "std": float(df[values_col].std()),
            "min": float(df[values_col].min()),
            "max": float(df[values_col].max()),
            "length": int(len(df))
        }
        
         # Save in context_info
        update_context('statistics', {'series_mean': stats["mean"], 'series_std': stats["std"], 'min': stats["min"], 'max': stats["max"], 'series_length': stats["length"]})


        # Add ADF test results
        adf_results = perform_adf_test(df[values_col].values)
        if adf_results:
            # Convert numpy types to Python native types
            stats["stationarity_test"] = {
                "test_statistic": float(adf_results["test_statistic"]),
                "p_value": float(adf_results["p_value"]),
                "critical_values": {k: float(v) for k, v in adf_results["critical_values"].items()},
                "is_stationary": bool(adf_results["is_stationary"])
            }

            update_context('stationarity', {
                "p_value": float(adf_results["p_value"]),
                "is_stationary": bool(adf_results["is_stationary"])
            })
        
        # Calculate ACF and PACF
        acf_pacf_results = calculate_acf_pacf(df[values_col].values)
        if acf_pacf_results:
            acf_pacf_results = {
                "acf": [float(x) for x in acf_pacf_results["acf"]],
                "pacf": [float(x) for x in acf_pacf_results["pacf"]],
                "conf_int": float(acf_pacf_results["conf_int"])
            }
        
        # Fit AutoARIMA
        logger.debug("Fitting AutoARIMA model")
        arima_results = fit_auto_arima(df[values_col].values)
        if arima_results:
            arima_results = {
                "predictions": [float(x) for x in arima_results["predictions"]],
                "order": [int(x) for x in arima_results["order"]],
                "aic": float(arima_results["aic"]),
                "info": str(arima_results["info"])
            }

            update_context('arima', {
                "order": [int(x) for x in arima_results["order"]],
                "aic": float(arima_results["aic"])
            })
        
        # Fit GARCH
        logger.debug("Fitting GARCH model")
        garch_results = fit_garch(df[values_col].values)
        if garch_results:
            garch_results = {
                "params": {k: float(v) for k, v in garch_results["params"].items()},
                "volatility": [float(x) for x in garch_results["volatility"]]
            }
        
        # Prepare the response
        response_data = {
            "timestamps": df[timestamp_col].dt.strftime('%Y-%m-%d').tolist(),
            "values": [float(x) for x in df[values_col].tolist()],
            "statistics": stats,
        }
        
        # Debug app.state

        #print(app.state.context_info)

        # Add results if available
        if acf_pacf_results:
            response_data["acf_pacf"] = acf_pacf_results
        if arima_results:
            response_data["arima"] = arima_results
        if garch_results:
            response_data["garch"] = garch_results
        
        logger.info(f"Analysis completed successfully for {filename}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@app.post("/forecast/{filename}")
async def generate_forecast(
    filename: str, 
    model_type: str = Query(..., description="Type of forecasting model to use"),
    forecast_periods: int = Query(..., description="Number of periods to forecast")
):
    logger.info(f"Starting forecast for file: {filename} with model {model_type} for {forecast_periods} periods")
   


    try:
        # Read and prepare data
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        df = pd.read_csv(file_path)
        
        # Identify columns
        timestamp_col = next((col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()), df.columns[0])
        values_col = next((col for col in df.columns if 'values' in col.lower() or 'sales' in col.lower()), df.columns[1])
        
        # Prepare data
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df[values_col] = pd.to_numeric(df[values_col], errors='coerce')
        df = df.dropna().sort_values(timestamp_col)
        
        # Split data
        train_df, val_df = split_data(df, timestamp_col, values_col)
        
        # Choose and fit model
        if model_type.lower() == 'prophet':
            results = fit_prophet(train_df, val_df, timestamp_col, values_col, forecast_periods)
        elif model_type.lower() == 'nhits':
            results = fit_nhits(train_df, val_df, timestamp_col, values_col, forecast_periods)
        elif model_type.lower() == 'informer':
            results = fit_informer(train_df, val_df, timestamp_col, values_col, forecast_periods)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        if results is None:
            raise HTTPException(status_code=500, detail="Error in forecast generation")
        
        # Add training and validation data to results
        results.update({
            'train_dates': train_df[timestamp_col].dt.strftime('%Y-%m-%d').tolist(),
            'train_values': train_df[values_col].tolist(),
            'val_dates': val_df[timestamp_col].dt.strftime('%Y-%m-%d').tolist(),
            'val_values': val_df[values_col].tolist()
        })
        
        update_context(f'forecast_{model_type}', {
            'metrics': results['metrics'],
            'forecast_dates': results['forecast_dates'],
            'forecast_predictions': results['forecast_predictions']
        })


        # Debug app.state
        print(app.state.context_info)

        return results
        
    except Exception as e:
        logger.error(f"Error in forecast generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.post("/save-file")
async def save_file(request: Request):
    try:
        # Get the content type and filename from headers
        content_type = request.headers.get("content-type", "")
        #filename = request.headers.get("x-file-name", "forecast.png")
        filename="forecast.png"
        # Read the binary content
        content = await request.body()
        
        # Define the path where the file will be saved
        #save_path = os.path.join(os.path.dirname(__file__), filename)
        save_path = os.path.join('./downloads', filename)
        
        # Save the file
        with open(save_path, "wb") as f:
            f.write(content)
        
        return Response(
            content=f"File {filename} saved successfully",
            media_type="text/plain",
            status_code=200
        )
        
    except Exception as e:
        return Response(
            content=f"Error saving file: {str(e)}",
            media_type="text/plain",
            status_code=500
        )

@app.post("/create-directory")
async def create_directory():
    try:
        # Create downloads directory in the same folder as the script
        downloads_dir = os.path.join(os.path.dirname(__file__), "downloads")
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)
        return Response(
            content="Directory created successfully",
            media_type="text/plain",
            status_code=200
        )
    except Exception as e:
        return Response(
            content=f"Error creating directory: {str(e)}",
            media_type="text/plain",
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
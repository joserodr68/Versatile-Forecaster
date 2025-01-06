import os
from typing import Dict, Type
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import pandas as pd
import numpy as np
import json

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'

class TimeSeriesInput(BaseModel):
    """Input for time series analysis."""
    context_info: Dict = Field(
        ..., description="Dictionary containing time series data and forecast information"
    )

class TimeSeriesAnalyzer(BaseTool):
    name: str = "time_series_analyzer"
    description: str = "Analyzes time series data and provides forecast insights"
    args_schema: Type[BaseModel] = TimeSeriesInput

    def _run(self, context_info: Dict) -> Dict:
        """Run time series analysis."""
        try:
            # Extract statistics
            stats = context_info['statistics']
            is_stationary = context_info['stationarity']['is_stationary']
            arima_order = context_info['arima']['order']

            # Process forecasts
            available_forecasts = []
            forecast_mapes = {}
            for model in ['prophet', 'nhits']:
                if f'forecast_{model}' in context_info:
                    available_forecasts.append(model)
                    forecast_mapes[model] = context_info[f'forecast_{model}']['metrics']['mape']

            if not available_forecasts:
                raise ValueError("No forecast data available")

            # Select best model
            best_forecast = min(forecast_mapes.items(), key=lambda x: x[1])[0]
            best_forecast_data = context_info[f'forecast_{best_forecast}']
            best_mape = forecast_mapes[best_forecast]

            # Analyze trend
            predictions = best_forecast_data['forecast_predictions']
            trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            trend_direction = "increasing" if trend > 0 else "decreasing"

            # Process forecasts
            forecast_df = pd.DataFrame({
                'date': pd.to_datetime(best_forecast_data['forecast_dates']),
                'value': predictions
            })

            # Handle values sequence
            totals = {}
            if context_info['series type'] == 'values sequence':
                totals = {
                    'weekly_totals': forecast_df.set_index('date').resample('W')['value'].sum().to_dict(),
                    'total_forecast': sum(predictions)
                }

            # ARIMA analysis
            arima_complexity = sum(arima_order)
            arima_analysis = {
                'complexity': 'low' if arima_complexity <= 2 else 'medium' if arima_complexity <= 4 else 'high',
                'differencing_needed': arima_order[1] > 0,
                'seasonality_indication': arima_order[2] > 1
            }

            # Volatility calculation
            volatility_metrics = {
                'coefficient_of_variation': stats['series_std'] / stats['series_mean'] * 100,
                'range_ratio': (stats['max'] - stats['min']) / stats['series_mean'] * 100
            }

            return {
                'forecast_info': {
                    'available_models': available_forecasts,
                    'selected_model': best_forecast,
                    'mape': best_mape,
                    'expected_accuracy': 100 - best_mape,
                    'trend': trend_direction,
                    'trend_magnitude': abs(trend)
                },
                'forecast_totals': totals if context_info['series type'] == 'values sequence' else None,
                'series_characteristics': {
                    'is_stationary': is_stationary,
                    'mean': stats['series_mean'],
                    'std': stats['series_std'],
                    'volatility': volatility_metrics
                },
                'arima_insights': {
                    'model_order': arima_order,
                    'analysis': arima_analysis,
                    'aic': context_info['arima']['aic']
                },
                'forecast_reliability': {
                    'confidence': 'high' if best_mape < 10 else 'medium' if best_mape < 20 else 'low',
                    'stationarity_support': is_stationary,
                    'volatility_impact': 'low' if volatility_metrics['coefficient_of_variation'] < 30 else 'medium' if volatility_metrics['coefficient_of_variation'] < 50 else 'high'
                }
            }
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

    async def _arun(self, context_info: Dict) -> Dict:
        """Async version not implemented."""
        raise NotImplementedError("Async execution not supported")


def create_report_templates():
    """Create templates for each section of the report."""
    templates = {
        'executive_summary': PromptTemplate(
            input_variables=["forecast_info"],
            template="""Based on the forecast information: Model Selection: {forecast_info} Provide a concise executive summary focusing on: 1. Selected forecast model and its accuracy 2. Key trend direction and magnitude 3. Overall forecast reliability Format the summary in professional business language with clear metrics."""
        ),
        'forecast_analysis': PromptTemplate(
            input_variables=["forecast_info", "forecast_totals"],
            template="""Analyze the following forecast data: Forecast Information: {forecast_info} Forecast Totals: {forecast_totals} Provide a detailed analysis including: 1. Trend analysis with specific numbers 2. Weekly patterns and total projections if available 3. Model performance metrics Present the analysis with specific numbers and clear business implications."""
        ),
        'reliability_assessment': PromptTemplate(
            input_variables=["forecast_reliability", "series_characteristics"],
            template="""Assess the forecast reliability based on: Reliability Metrics: {forecast_reliability} Series Characteristics: {series_characteristics} Provide: 1. Confidence level assessment 2. Statistical validity analysis 3. Key factors affecting reliability Use specific metrics and clear technical explanations."""
        ),
        'business_implications': PromptTemplate(
            input_variables=["forecast_info", "forecast_reliability"],
            template="""Based on: Forecast Information: {forecast_info} Reliability Assessment: {forecast_reliability} Provide: 1. Key business implications 2. Actionable recommendations 3. Risk considerations Focus on practical, implementable insights for business decision-makers."""
        )
    }
    return templates

def format_weekly_totals(weekly_totals):
    """Format weekly totals into a markdown table."""
    if not weekly_totals:
        return "No weekly totals available."
    table = "| Week Ending | Forecasted Total |\n|-------------|------------------|\n"
    for date, value in weekly_totals.items():
        formatted_date = pd.Timestamp(date).strftime('%Y-%m-%d')
        formatted_value = f"{value:,.2f}"
        table += f"| {formatted_date} | {formatted_value} |\n"
    return table

def generate_time_series_report(context_info: dict) -> str:
    """Generate a complete time series analysis report."""
    # Initialize components
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    analyzer = TimeSeriesAnalyzer()
    templates = create_report_templates()

    # Add output parser to get clean markdown
    class MarkdownOutputParser:
        def parse(self, message):
            # Extract just the content from the message, removing any metadata
            if hasattr(message, 'content'):
                return message.content
            # If it's a string with metadata, extract just the content before any metadata markers
            if isinstance(message, str):
                # Split on common metadata markers and take the first part
                content = message.split('additional_kwargs')[0]
                content = content.split('response_metadata')[0]
                content = content.strip()
                return content
            return str(message)

    markdown_parser = MarkdownOutputParser()

    # Run analysis
    analysis_result = analyzer.run({"context_info": context_info})

    # Modify chains to include the parser
    chains = {
        name: (template | llm | markdown_parser.parse)
        for name, template in templates.items()
    }

    # Generate report sections
    report_sections = {}
    for name, chain in chains.items():
        report_sections[name] = chain.invoke({
            "forecast_info": analysis_result["forecast_info"],
            "forecast_totals": analysis_result["forecast_totals"],
            "forecast_reliability": analysis_result["forecast_reliability"],
            "series_characteristics": analysis_result["series_characteristics"]
        })

    # Format weekly totals if available
    weekly_totals_table = ""
    if analysis_result.get("forecast_totals") and analysis_result["forecast_totals"].get("weekly_totals"):
        weekly_totals_table = "\n\n### Weekly Totals\n" + format_weekly_totals(
            analysis_result["forecast_totals"]["weekly_totals"]
        )

    def format_dict_as_markdown(d: dict) -> str:
        """Convert a dictionary to markdown-formatted text."""
        return '\n'.join(f"- {key}: {value}" for key, value in d.items())

    # Create technical details section with markdown formatting
    technical_details = f"""### Model Configuration
{format_dict_as_markdown(analysis_result['arima_insights'])}

### Volatility Metrics
{format_dict_as_markdown(analysis_result['series_characteristics']['volatility'])}

### Forecast Metrics
- MAPE: {analysis_result['forecast_info']['mape']}
- Expected Accuracy: {analysis_result['forecast_info']['expected_accuracy']}
"""

    # Compile final report
    final_report = f"""# Time Series Analysis Report

## Executive Summary
{report_sections['executive_summary']}

## Forecast Analysis
{report_sections['forecast_analysis']}
{weekly_totals_table}

## Reliability Assessment
{report_sections['reliability_assessment']}

## Business Implications
{report_sections['business_implications']}

## Technical Details
{technical_details}
"""
    return final_report


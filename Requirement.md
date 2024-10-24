# Olive Production Dataset Generation for Algeria


## Requirements

Before running the program, make sure the following are installed:

- **Python 3.x**: The program is written in Python and requires version 3.6 or later.
- **Python Libraries**: The script uses standard Python libraries, so no external dependencies are needed.

## Installation

1. **Verify that Python is installed**:
    - Make sure Python 3 is installed on your machine by running the following command in your terminal or command prompt:
      ```bash
      python --version
      ```

2. **Clone the repository**:
    - Clone this GitHub repository to your local machine:
      ```bash
      git clone https://github.com/your-username/olive-production-dataset.git
      cd olive-production-dataset
      ```

3. **No additional libraries required**:
    - This script only uses standard Python libraries such as `csv` and `random`, so no extra installation is needed.

## Usage

To generate a CSV file with simulated olive production data, follow these steps:

1. **Run the script**:
    - Execute the Python script `generate_olive_dataset.py` using the following command:
      ```bash
      python generate_olive_dataset.py
      ```

2. **Output**:
    - The script will generate a CSV file called `olive_production_dataset.csv` in the current directory. This file contains 500 rows of simulated data for olive production in Algeria.

## Dataset Structure

The generated CSV file contains the following columns:

| Feature           | Description                                      |
|-------------------|--------------------------------------------------|
| Region            | The region where olives are cultivated           |
| Year              | Harvest year                                     |
| Month             | Harvest month                                    |
| Area_Planted      | Planted area in hectares                         |
| Olive_Variety     | Type of olives grown                             |
| Irrigation_Method | Irrigation method used                           |
| Soil_Type         | Type of soil                                     |
| Fertilizer_Used   | Amount of fertilizer used (kg/ha)                |
| Pesticides_Used   | Amount of pesticides used (kg/ha)                |
| Tmin              | Average minimum temperature (°C)                 |
| Tmax              | Average maximum temperature (°C)                 |
| Tmean             | Average temperature (°C)                         |
| Precipitation     | Total monthly precipitation (mm)                 |
| Sunshine_Hours    | Monthly sunshine hours                           |
| Labor_Cost        | Labor cost per hectare (USD)                     |
| Market_Price      | Selling price per kg (USD)                       |
| Production        | Olive production (tonnes)                        |

## Example Output

After execution, the generated CSV file will look like this:

```csv
Region,Year,Month,Area_Planted,Olive_Variety,Irrigation_Method,Soil_Type,Fertilizer_Used,Pesticides_Used,Tmin,Tmax,Tmean,Precipitation,Sunshine_Hours,Labor_Cost,Market_Price,Production
Algiers,2019,10,150.32,Chemlal,Drip,Loamy,100.5,30.1,10.4,32.5,21.45,90.3,280.5,500,3.2,350.8
Oran,2018,3,200.50,Sigoise,Rain-fed,Clay,120.3,15.2,12.5,28.4,20.45,110.5,310.4,400,4.1,450.5
...

import pandas as pd
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def convert_csvs_to_parquet(input_dir, output_dir):
    """
    Convert all CSV files in input_dir to parquet files in output_dir
    
    Args:
        input_dir (str): Directory containing CSV files
        output_dir (str): Directory where parquet files will be saved
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files to convert")
    
    # Process each file
    for csv_file in tqdm(csv_files, desc="Converting files"):
        try:
            # Construct file paths
            csv_path = os.path.join(input_dir, csv_file)
            parquet_file = csv_file.replace('.csv', '.parquet')
            parquet_path = os.path.join(output_dir, parquet_file)
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Convert to parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path)
            
        except Exception as e:
            print(f"Error converting {csv_file}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_directories = [
        './data/FAB/FAB_A_HandRelative_Motion',
        #'./data/FAB/FAB_B_Modified_Motion_Pause',
        #'./data/FAB'  # For metadata.csv
    ]
    
    for input_dir in input_directories:
        output_dir = './data/FAB/FAB_A_HandRelative_Motion_PQ'
        print(f"\nProcessing directory: {input_dir}")
        convert_csvs_to_parquet(input_dir, output_dir)
        
    print("\nConversion complete!") 
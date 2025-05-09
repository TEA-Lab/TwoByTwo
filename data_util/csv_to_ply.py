import pandas as pd
import argparse

# Function to read the CSV file and remove duplicates
def read_csv_remove_duplicates(input_csv):
    df = pd.read_csv(input_csv, header=None, names=["x", "y", "z"])
    # df = df.drop_duplicates()
    return df

# Function to write the DataFrame to a PLY file
def write_ply(df, output_ply):
    with open(output_ply, 'w+') as ply_file:
        ply_file.write('ply\n')
        ply_file.write('format ascii 1.0\n')
        ply_file.write(f'element vertex {len(df)}\n')
        ply_file.write('property float x\n')
        ply_file.write('property float y\n')
        ply_file.write('property float z\n')
        ply_file.write('property uchar red\n')
        ply_file.write('property uchar green\n')
        ply_file.write('property uchar blue\n')
        ply_file.write('end_header\n')
        for _, row in df.iterrows():
            ply_file.write(f'{row["x"]} {row["y"]} {row["z"]} 255 0 0\n')

# Main function to convert CSV to PLY
def convert_csv_to_ply(input_csv, output_ply):
    df = read_csv_remove_duplicates(input_csv)
    write_ply(df, output_ply)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to PLY")
    parser.add_argument('--input_csv', default='points.csv', type=str)
    parser.add_argument('--output_ply', default='points.ply', type=str)
    args= parser.parse_args()
    convert_csv_to_ply(args.input_csv, args.output_ply)

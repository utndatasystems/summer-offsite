# summer-offsite
Why Compress When You Have an Oracle? LLM-Powered Text Compression

## Project Structure
```
├── data/                    # Stores raw text data
├── tmp/                     # Temporary files during compression/decompression
├── test.py                  # Test script to test compression/decompression
├── compressor.py            # Main script for compression and decompression
├── datasets_download.py     # Script to download datasets
├── results.json             # Output file recording metrics (size, time)
├── datasets_info.json       # JSON config listing dataset metadata
├── LICENSE                  # Open source license for the project
├── README.md                # README File
```

## Dataset Download
```
python3 datasets_download.py
```

## Testing With text8
```
python3 main.py --mode compress --input_path data/text8 --batch_size 16
```
The results will be updated in `results.json`. The performance metrics are:
- `total_compression_time`: tokenization + LLM inference + AC
- `tokenize_time`: tokenization
- `compression_time`: LLM inference + AC
- `inference_time`: LLM inference
- `ac_time`: AC
- `data_copy_time`: Time spent moving data between GPU and CPU
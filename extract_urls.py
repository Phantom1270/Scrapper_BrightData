import json
import csv
import os

def main():
    phase2_file = r"d:\New folder\Scrapper_BrightData\phase2\phase2_output.json"
    output_txt = r"d:\New folder\Scrapper_BrightData\brightdata_urls.txt"
    output_csv = r"d:\New folder\Scrapper_BrightData\brightdata_urls.csv"
    
    if not os.path.exists(phase2_file):
        print(f"Error: Could not find {phase2_file}")
        return

    with open(phase2_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    template_map = data.get("template_map", {})
    
    # Collect a mix of URLs
    all_urls = []
    for urls in template_map.values():
        all_urls.extend(urls)
        
    # Limit to exactly 50
    sample_urls = all_urls[:50]
    
    # Save as TXT (one URL per line)
    with open(output_txt, "w", encoding="utf-8") as f:
        for url in sample_urls:
            f.write(url + "\n")
            
    # Save as CSV (with a header 'url', which Bright Data usually prefers)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        for url in sample_urls:
            writer.writerow([url])
            
    print(f"Successfully extracted {len(sample_urls)} URLs.")
    print(f"Saved to:\n - {output_txt}\n - {output_csv}")

if __name__ == "__main__":
    main()

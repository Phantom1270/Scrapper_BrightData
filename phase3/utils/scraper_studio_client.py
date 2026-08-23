import subprocess
import json
import re
from typing import Optional

class ScraperStudioCLI:
    """Wrapper for the @brightdata/cli to interact with Scraper Studio."""
    
    def __init__(self, npx_command="npx -y -p @brightdata/cli bdata"):
        self.cmd_prefix = npx_command

    def create_scraper(self, url: str, prompt: str) -> Optional[str]:
        print(f"    [CLI] Creating scraper for {url}...")
        cmd = f'{self.cmd_prefix} scraper create "{url}" "{prompt}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        
        if result.returncode != 0:
            print(f"    [CLI Error] Failed to create scraper: {result.stderr}")
            return None
            
        # Extract Collector ID (starts with c_)
        match = re.search(r'(c_[a-zA-Z0-9]+)', result.stdout)
        if match:
            return match.group(1)
            
        # If regex fails but it succeeded, it might just print the ID
        out = result.stdout.strip()
        if out.startswith("c_"):
            return out
            
        print(f"    [CLI Warning] Could not parse Collector ID from output: {out}")
        return None
        
    def run_scraper(self, collector_id: str, url: str) -> Optional[dict]:
        cmd = f'{self.cmd_prefix} scraper run {collector_id} "{url}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        
        if result.returncode != 0:
            return None
            
        try:
            # Try direct parse first (cleanest case)
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            pass

        # CLI may prefix the JSON with spinner/log lines.
        # Scan for the first valid JSON object or array by trying each candidate.
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            idx = result.stdout.find(start_char)
            while idx != -1:
                candidate = result.stdout[idx:]
                # Find the matching closing bracket
                end_idx = candidate.rfind(end_char)
                if end_idx != -1:
                    try:
                        return json.loads(candidate[:end_idx + 1])
                    except json.JSONDecodeError:
                        pass
                idx = result.stdout.find(start_char, idx + 1)

        # If it's not valid JSON, return the raw text wrapped in a dict
        return {"raw_output": result.stdout.strip()}

    def heal_scraper(self, collector_id: str, prompt: str) -> bool:
        print(f"    [CLI] Healing scraper {collector_id}...")
        cmd = f'{self.cmd_prefix} scraper heal {collector_id} "{prompt}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        
        if result.returncode != 0:
            print(f"    [CLI Error] Heal failed: {result.stderr}")
            return False
            
        print(f"    [CLI] Approving heal for {collector_id}...")
        cmd_approve = f'{self.cmd_prefix} scraper approve {collector_id}'
        res_approve = subprocess.run(cmd_approve, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        
        return res_approve.returncode == 0

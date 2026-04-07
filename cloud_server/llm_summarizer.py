import json
import os

# Point this to where the Edge Node saved the batch JSON
JSON_FILE_PATH = "../data/master_metadata_log.json"

def generate_auditor_report_prompt(json_path):
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Start building the prompt context
    prompt = f"""
You are an expert Loss Prevention Auditor for a Gold Loan branch. 
Review the following automated Edge AI telemetry data and generate a concise Security Incident Report.

--- SYSTEM OVERVIEW ---
Total Videos Processed: {data.get('total_videos_processed', 0)}

--- CRITICAL INCIDENTS LOG ---
"""
    
    incident_count = 0

    # 2. Parse the JSON and filter for ONLY actionable intelligence
    for video in data.get('processed_videos', []):
        video_name = video.get('filename', 'Unknown')
        has_incidents = False
        video_log = f"\nVideo Source: {video_name}\n"

        for chunk in video.get('chunks', []):
            timestamp = f"00:00:{chunk['chunk_id'] // 2:02d}" # Rough timestamp calculation
            
            # Extract Suspicous Actions
            for track in chunk.get('tracking_data', []):
                if track['status'] == "SUSPICIOUS":
                    video_log += f"[{timestamp}] ALERT: ID {track['id']} flagged for '{track['action']}'.\n"
                    has_incidents = True
                    incident_count += 1
            
            # Extract Proximity Violations
            for alert in chunk.get('proximity_flags', []):
                video_log += f"[{timestamp}] PROXIMITY: {alert}\n"
                has_incidents = True
                incident_count += 1

        # Only add this video to the prompt if something bad actually happened
        if has_incidents:
            prompt += video_log
            
    if incident_count == 0:
        prompt += "\nNo suspicious activity or proximity violations detected across the batch.\n"

    # 3. Add the final instructions for the LLM
    prompt += """
--- INSTRUCTIONS ---
Based on the log above, please provide:
1. An Executive Summary of the security posture.
2. A bulleted list of specific High-Risk Incidents (if any).
3. Recommendations for branch staff.
"""
    return prompt

if __name__ == "__main__":
    print("Parsing Edge Node JSON...\n")
    final_prompt = generate_auditor_report_prompt(JSON_FILE_PATH)
    
    if final_prompt:
        print("=== PROMPT READY TO SEND TO LLM ===")
        print(final_prompt)
        
        # TODO: Pass `final_prompt` into your preferred LLM API (e.g., google.generativeai or openai)
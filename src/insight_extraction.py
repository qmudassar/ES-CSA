# Description: Module for extracting Consumer/General Insights.

import json
import re
from typing import List
from datetime import datetime
from langchain_core.documents import Document


# Extracting Consumer Insights

def compute_consumer_insights(docs: List[Document]) -> List[str]:
    facts = []
    summary = {}

    for doc in docs:
        metadata = doc.metadata
        section = metadata.get("section")
        content = doc.page_content.strip()

        if section == "user_profile":
            name = city = acc_type = "Unknown"
            match = re.search(r"User\s+(.+?)\s+from\s+([A-Za-z\s]+?)\s+is\s+on\s+a\s+(\w+)\s+plan", content)
            if match:
                name, city, acc_type = match.groups()
            summary.update({"Name": name, "City": city, "Account Type": acc_type})
            facts += [
                f"The user's name is {name}.",
                f"The user is located in {city}.",
                f"The user has a {acc_type} account."
            ]

        elif section == "cdrs":
            lines = content.split("\n")[1:]
            total_data = total_sms = total_voice = 0
            data_charge = sms_charge = voice_charge = total_charge = 0

            facts.append("Recent CDR logs:")
            for line in lines[:5]:
                line = re.sub(
                    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
                    lambda m: datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y at %I:%M %p"),
                    line
                )
                facts.append(f"- {line.strip()}")

                match = re.search(r"used\s+(\d+)\s+(\w+)\s+resources.*?charged\s+(\d+)\s+PKR", line)
                if match:
                    amount, resource, charge = match.groups()
                    amount, charge = int(amount), int(charge)
                    total_charge += charge
                    if resource.lower() == "data":
                        total_data += amount
                        data_charge += charge
                    elif resource.lower() == "sms":
                        total_sms += amount
                        sms_charge += charge
                    elif resource.lower() == "voice":
                        total_voice += amount
                        voice_charge += charge

            txn_count = len(lines)
            summary.update({
                "Total Data MB": total_data,
                "Total SMS": total_sms,
                "Total Voice Min": total_voice,
                "Data Charges": data_charge,
                "SMS Charges": sms_charge,
                "Voice Charges": voice_charge,
                "Total Charges": total_charge,
                "Resource Transactions": txn_count
            })
            facts += [
                f"Total Data Consumed: {total_data} MB.",
                f"Total SMS Sent: {total_sms}.",
                f"Total Voice Minutes: {total_voice} minutes.",
                f"Data Charges: {data_charge} PKR.",
                f"SMS Charges: {sms_charge} PKR.",
                f"Voice Charges: {voice_charge} PKR.",
                f"Total Resource Charge: {total_charge} PKR over {txn_count} transactions."
            ]

        elif section == "purchases":
            lines = content.split("\n")[1:]
            total_spent = 0
            facts.append("Recent Purchase Logs:")
            for line in lines[:5]:
                line = re.sub(
                    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
                    lambda m: datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y at %I:%M %p"),
                    line
                )
                facts.append(f"- {line.strip()}")
                match = re.search(r"spent\s+(\d+)\s+PKR", line)
                if match:
                    total_spent += int(match.group(1))

            summary.update({
                "Total Purchases": len(lines),
                "Total Purchase Charge": total_spent
            })
            facts += [
                f"Total Purchases: {len(lines)}.",
                f"Total Spent on Purchases: {total_spent} PKR."
            ]

        elif section == "tickets":
            lines = content.split("\n")[1:]
            total_tickets = len(lines)
            recent = lines[:5]
            facts.append("Recent Support History:")
            for line in recent:
                line = re.sub(
                    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
                    lambda m: datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y at %I:%M %p"),
                    line
                )
                match = re.search(r"Ticket\s+(\w+)", line)
                ticket_id = match.group(1) if match else "Unknown"
                facts.append(f"- {ticket_id}: {line.strip()}")

            latest_status = "Unknown"
            for line in reversed(lines):
                match = re.search(r"Ticket\s+(\w+).*?resolved on\s+([\d\-T:]+).*?resolution:\s+(.*?)\.", line)
                if match:
                    ticket_id, resolved_time, resolution = match.groups()
                    latest_status = "Resolved"
                    try:
                        formatted_time = datetime.strptime(resolved_time, "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y at %I:%M %p")
                    except Exception:
                        formatted_time = resolved_time
                    facts += [
                        f"Latest Ticket ID: {ticket_id}.",
                        f"Latest Ticket Status: {latest_status}.",
                        f"Latest Ticket Resolved Time: {formatted_time}.",
                        f"Latest Ticket Resolution: {resolution.strip()}."
                    ]
                    break

            summary["Total Tickets"] = total_tickets
            facts.append(f"Total Support Tickets Raised: {total_tickets}.")

    return facts

# Extracting General Insights

def compute_general_insights(docs: List[dict]) -> List[str]:

    facts = []
    summary = {}

    for doc in docs:
        metadata = doc.metadata
        subcat = metadata.get("subcategory")
        lines = doc.page_content.strip().split("\n")

        if subcat == "Regional Popularity":
            region_user_map = {}
            for line in lines:
                match = re.search(r"The city of (.+?) has (\d+) active users", line)
                if match:
                    city, count = match.groups()
                    region_user_map[city] = int(count)
                    facts.append(f"{city} has {count} active users.")

            most_city = max(region_user_map, key=region_user_map.get)
            least_city = min(region_user_map, key=region_user_map.get)

            facts.append(f"{most_city} has the most users: {region_user_map[most_city]}.")
            facts.append(f"{least_city} has the least users: {region_user_map[least_city]}.")

            summary["Regional Popularity"] = region_user_map

        elif subcat == "User Type Distribution":
            for line in lines:
                match = re.search(r"There are (\d+) (Postpaid|Prepaid) users", line)
                if match:
                    count, user_type = match.groups()
                    facts.append(f"There are {count} {user_type.lower()} users in the network.")
            summary["User Type Distribution"] = lines

        elif subcat == "Regional User Type Distribution":
            for line in lines:
                match = re.search(r"In (.+?), there are (\d+) postpaid users and (\d+) prepaid users", line)
                if match:
                    city, postpaid, prepaid = match.groups()
                    postpaid = int(postpaid)
                    prepaid = int(prepaid)
                    more_popular = "postpaid" if postpaid > prepaid else "prepaid"
                    facts.append(f"In {city}, there are {postpaid} postpaid users and {prepaid} prepaid users.")
                    facts.append(f"{more_popular.capitalize()} is more popular in {city}.")
            summary["Regional User Type Distribution"] = lines

        elif subcat == "Most Common Ticket Categories":
            category_map = {}
            for line in lines:
                match = re.search(r"The '(.+?)' category has (\d+) support tickets", line)
                if match:
                    category, count = match.groups()
                    count = int(count)
                    category_map[category] = count
                    facts.append(f"{category} tickets: {count} logged.")

            most_common = max(category_map, key=category_map.get)
            facts.append(f"The most common ticket category is '{most_common}' with {category_map[most_common]} tickets.")
            summary["Most Common Ticket Categories"] = category_map

        elif subcat == "Average Resolution Time Per Ticket Category":
            for line in lines:
                match = re.search(r"The average resolution time for '(.+?)' tickets is ([\d.]+) hours", line)
                if match:
                    category, hours = match.groups()
                    facts.append(f"Average resolution time for {category} tickets: {hours} hours.")
            summary["Resolution Times"] = lines

    return facts
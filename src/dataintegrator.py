# Description: This file contains the class to integrate user data from the different sources and store it in the database in nested and structured form.

import pandas as pd

class DataIntegrator:
    def __init__(self, userprofile, cdrs, purchases, tickets):
        """
        Initializes the DataIntegrator with required datasets.
        """
        self.userprofile = userprofile
        self.cdrs = cdrs
        self.purchases = purchases
        self.tickets = tickets

    def restructure_cdrs(self, msisdn):
        """
        Extracts CDR entries for a given MSISDN and structures them as a list of dictionaries.
        """
        user_cdrs = self.cdrs[self.cdrs["MSISDN"] == msisdn]
        cdrs_entries = []

        for _, row in user_cdrs.iterrows():
            cdrs_entries.append({
                "Amount_Charged": row["Amount_Charged"],
                "Resource_Value": row["Resource_Value"],
                "Resource_Type": row["Resource_Type"],
                "Datetime_Charged": row["Datetime_Charged"]
            })

        return cdrs_entries if cdrs_entries else None

    def restructure_purchases(self, msisdn):
        """
        Extracts purchase entries for a given MSISDN and structures them as a list of dictionaries.
        """
        user_purchases = self.purchases[self.purchases["MSISDN"] == msisdn]
        purchase_entries = []

        for _, row in user_purchases.iterrows():
            purchase_entries.append({
                "Datetime": row["Datetime"],
                "Amount": row["Amount"],
                "Data_Browsing_Allowance": row["Data_Browsing_Allowance"],
                "SMS_Allowance": row["SMS_Allowance"],
                "Voice_On-Net_Allowance": row["Voice_On-Net_Allowance"],
                "Voice_Off-Net_Allowance": row["Voice_Off-Net_Allowance"],
                "Data_Social_Allowance": row["Data_Social_Allowance"]
            })

        return purchase_entries if purchase_entries else None

    def restructure_tickets(self, msisdn):
        """
        Extracts ticket entries for a given MSISDN and structures them as a list of dictionaries.
        """
        user_tickets = self.tickets[self.tickets["MSISDN"] == msisdn]
        ticket_entries = []

        for _, row in user_tickets.iterrows():
            ticket_entries.append({
                "Ticket_ID": row["Ticket_ID"],
                "Log_Time": row["Log_Time"],
                "Resolution_Time": row["Resolution_Time"],
                "Category": row["Category"],
                "Description": row["Description"],
                "Resolution": row["Resolutions"]
            })

        return ticket_entries if ticket_entries else None

    def integrate_data(self):
        """
        Integrates user data with nested CDR, Purchase, and Ticket records.
        """
        nested_data = []

        for _, row in self.userprofile.iterrows():
            msisdn = row["MSISDN"]
            structured_entry = {
                "MSISDN": msisdn,
                "Name": row["Name"],
                "City": row["City"],
                "User_Type": row["User_Type"],
                "CDRS": self.restructure_cdrs(msisdn),
                "Purchases": self.restructure_purchases(msisdn),
                "Tickets": self.restructure_tickets(msisdn),
            }
            nested_data.append(structured_entry)

        return pd.DataFrame(nested_data)

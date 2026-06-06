"""
Synthetic Support Ticket Generator

Generates realistic customer support tickets with various categories and priorities
for training and testing the classification model.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd


class TicketGenerator:
    """
    Generates synthetic support tickets with realistic text patterns.
    
    Categories:
        - Billing: Payment issues, invoices, refunds
        - Technical: Product bugs, errors, technical problems
        - Account: Login issues, profile, settings
        - Feature Request: New feature suggestions
        - General Inquiry: General questions and information
    
    Priorities:
        - Critical: System down, security issues
        - High: Major functionality broken
        - Medium: Inconvenience but workaround exists
        - Low: Minor issues, questions
    """
    
    # Category templates with subject and description patterns
    TEMPLATES = {
        "Billing": {
            "subjects": [
                "Charged twice for subscription",
                "Invoice discrepancy for {month}",
                "Refund request for order #{order_id}",
                "Payment failed but account charged",
                "Need invoice for tax purposes",
                "Subscription renewal issue",
                "Credit card declined incorrectly",
                "Promo code not applied to order",
                "Incorrect billing amount",
                "Cancel subscription and refund",
                "Payment method update not working",
                "Billing cycle change request",
                "Dispute charge on my account",
                "Missing receipt for payment",
                "Auto-renewal disabled but still charged"
            ],
            "descriptions": [
                "I was charged ${amount} on {date} but my subscription is only ${sub_amount}/month. Please investigate and refund the difference.",
                "My invoice #{invoice_id} shows an incorrect amount. I should have been charged ${sub_amount} but was billed ${amount}.",
                "I need a refund for order #{order_id} placed on {date}. The product did not meet my expectations and I would like my money back.",
                "My payment failed with error code {error_code} but I can see the charge on my bank statement. Please help resolve this.",
                "I need a proper invoice with tax details for my purchase on {date}. This is required for my company's expense report.",
                "My subscription was supposed to renew at the promotional rate of ${sub_amount} but I was charged ${amount}.",
                "My credit card is valid but keeps getting declined. I've verified with my bank that the card is active.",
                "I applied promo code '{promo_code}' during checkout but the discount wasn't applied to my order.",
                "The billing amount on my recent charge doesn't match what was displayed during checkout. I was expecting ${sub_amount}.",
                "I want to cancel my subscription effective immediately and request a prorated refund for the unused portion.",
                "I'm trying to update my payment method but the system keeps showing an error. I've tried multiple cards.",
                "I would like to change my billing cycle from monthly to annual to get the discount. How do I do this?",
                "I don't recognize the charge of ${amount} on {date}. I did not authorize this transaction.",
                "I made a payment on {date} but never received a receipt or confirmation email.",
                "I disabled auto-renewal last month but was still charged ${amount} today. Please refund and cancel."
            ],
            "priority_weights": {"Critical": 0.1, "High": 0.3, "Medium": 0.4, "Low": 0.2}
        },
        "Technical": {
            "subjects": [
                "Application crashes on startup",
                "Error 500 when accessing dashboard",
                "Bug: Data not saving correctly",
                "Performance issues with large files",
                "Integration with {service} broken",
                "Cannot export data to {format}",
                "API returning incorrect response",
                "Mobile app not syncing",
                "Search functionality not working",
                "Page loading extremely slow",
                "SSL certificate error",
                "Database connection timeout",
                "Feature stopped working after update",
                "Memory leak in application",
                "File upload failing consistently"
            ],
            "descriptions": [
                "The application crashes immediately after I click the launch button. I'm running {os} version {version}. Error message: '{error_msg}'.",
                "Every time I try to access the dashboard, I get a 500 Internal Server Error. This started happening {time_ago}.",
                "When I save my changes, they appear to save but are lost when I refresh the page. This affects {feature} specifically.",
                "The application becomes extremely slow when working with files larger than {size}MB. It sometimes freezes completely.",
                "The integration with {service} stopped working after the last update. API calls are returning {error_code} errors.",
                "I cannot export my data to {format} format. The export button is grayed out or clicking it does nothing.",
                "The API endpoint /api/{endpoint} is returning {error_code} with message: '{error_msg}'. This worked fine yesterday.",
                "My mobile app hasn't synced with the desktop version for {days} days. I've tried reinstalling but the issue persists.",
                "The search feature returns no results even for terms I know exist in my data. This is affecting my productivity.",
                "Pages are taking over {seconds} seconds to load. My internet connection is fine as other sites load quickly.",
                "I'm getting an SSL certificate error when trying to connect. The error says '{error_msg}'.",
                "Getting 'Database connection timeout' errors frequently. The application becomes unusable during these periods.",
                "After the update to version {version}, the {feature} feature completely stopped working. Please fix urgently.",
                "The application's memory usage keeps increasing until it crashes. Task manager shows it using over {memory}GB.",
                "File uploads fail at {percent}% every time. I've tried different file sizes and types but same result."
            ],
            "priority_weights": {"Critical": 0.25, "High": 0.35, "Medium": 0.3, "Low": 0.1}
        },
        "Account": {
            "subjects": [
                "Cannot login to my account",
                "Password reset not working",
                "Account locked after failed attempts",
                "Two-factor authentication issue",
                "Need to change email address",
                "Account deletion request",
                "Profile picture not updating",
                "Username change request",
                "Account merge request",
                "Session keeps expiring",
                "Forgot security questions",
                "Account hacked - urgent",
                "Cannot verify phone number",
                "Account downgrade request",
                "Privacy settings not saving"
            ],
            "descriptions": [
                "I cannot login to my account. I'm entering the correct password but getting 'Invalid credentials' error.",
                "I requested a password reset {time_ago} but haven't received the email. I've checked spam folder too.",
                "My account was locked after multiple failed login attempts. I was trying to remember my password. Please unlock.",
                "My authenticator app was reset and I lost my 2FA codes. I can't access my account now. Need help urgently.",
                "I need to update my email address from {old_email} to {new_email}. How can I do this?",
                "I would like to permanently delete my account and all associated data as per GDPR requirements.",
                "I uploaded a new profile picture {time_ago} but it still shows my old one. I've tried clearing cache.",
                "I want to change my username from '{old_username}' to '{new_username}'. Is this possible?",
                "I have two accounts and would like to merge them into one. Old account: {old_email}, keep: {new_email}.",
                "My session expires every few minutes and I have to keep logging in. This is very frustrating.",
                "I forgot the answers to my security questions and can't proceed with account recovery.",
                "I believe my account has been compromised. I'm seeing activity I didn't authorize. Please help immediately!",
                "I'm trying to add my phone number for verification but keep getting 'Invalid number' error.",
                "I want to downgrade from Premium to Free plan. How do I proceed and when does it take effect?",
                "I change my privacy settings but they revert back to default when I log in again."
            ],
            "priority_weights": {"Critical": 0.15, "High": 0.25, "Medium": 0.35, "Low": 0.25}
        },
        "Feature Request": {
            "subjects": [
                "Request: Dark mode support",
                "Suggestion: Keyboard shortcuts",
                "Feature idea: Bulk export option",
                "Enhancement: Better mobile experience",
                "Request: API webhook support",
                "Suggestion: Custom dashboard widgets",
                "Feature: Collaborative editing",
                "Enhancement: Advanced search filters",
                "Request: Email notification preferences",
                "Suggestion: Integration with {service}",
                "Feature: Offline mode support",
                "Enhancement: Report customization",
                "Request: Multi-language support",
                "Suggestion: Audit log feature",
                "Feature: Template library"
            ],
            "descriptions": [
                "Would love to see a dark mode option. Many of us work late and the bright interface is hard on the eyes.",
                "It would be great to have keyboard shortcuts for common actions like save (Ctrl+S), new (Ctrl+N), etc.",
                "Please add ability to bulk export multiple items at once. Currently I have to export one by one which is tedious.",
                "The mobile experience could be improved. Specifically, {feature} is difficult to use on small screens.",
                "We need webhook support for the API to receive real-time updates when events occur in the system.",
                "Would be helpful to customize dashboard with widgets showing metrics that matter most to our team.",
                "Real-time collaborative editing would be amazing. Like Google Docs where multiple users can edit simultaneously.",
                "The search could use advanced filters like date range, created by, status, etc. to narrow results.",
                "Please add granular email notification settings. I want to choose which events trigger emails.",
                "Integration with {service} would be incredibly useful for our workflow. Many users would benefit.",
                "Offline mode would be a game changer. We often work in areas with poor connectivity.",
                "Report customization options are limited. Would like to add/remove columns and change formatting.",
                "Multi-language support would help our international team. Especially {language} translation.",
                "An audit log showing who made what changes and when would help with compliance requirements.",
                "A library of templates that users can share and reuse would save a lot of time."
            ],
            "priority_weights": {"Critical": 0.02, "High": 0.08, "Medium": 0.3, "Low": 0.6}
        },
        "General Inquiry": {
            "subjects": [
                "Question about pricing plans",
                "How to get started?",
                "Documentation for API",
                "Enterprise plan inquiry",
                "Training resources available?",
                "Data security question",
                "Compliance certification inquiry",
                "Partnership opportunity",
                "How does {feature} work?",
                "Comparison with competitors",
                "System requirements question",
                "SLA information request",
                "Bulk discount inquiry",
                "Reseller program question",
                "Product roadmap inquiry"
            ],
            "descriptions": [
                "I'm interested in your product but confused about the pricing tiers. Can you explain the differences?",
                "I just signed up and not sure where to start. Are there any getting started guides or tutorials?",
                "I'm a developer looking to integrate your API. Where can I find comprehensive documentation?",
                "We're a large company with {employees}+ employees. Do you offer enterprise plans with custom features?",
                "Are there any training materials, webinars, or certification programs available for your product?",
                "Before we onboard, I need to understand your data security practices. Do you have SOC2 compliance?",
                "Does your platform comply with {compliance} requirements? We're in a regulated industry.",
                "We're interested in exploring partnership opportunities. Who should we contact to discuss?",
                "I'm not sure how to use {feature}. Can you explain how it works or point me to documentation?",
                "How does your product compare to {competitor}? What are your key differentiators?",
                "What are the minimum system requirements to run your application? We're planning a deployment.",
                "Can you provide information about your SLA guarantees? Specifically uptime and response times.",
                "We need over {quantity} licenses. Do you offer volume discounts for larger purchases?",
                "Do you have a reseller or partner program? We'd like to offer your product to our clients.",
                "What features are planned for the next year? Is there a public roadmap we can review?"
            ],
            "priority_weights": {"Critical": 0.01, "High": 0.09, "Medium": 0.4, "Low": 0.5}
        }
    }
    
    # Placeholder data for template filling
    PLACEHOLDERS = {
        "month": ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"],
        "service": ["Slack", "Salesforce", "Jira", "Zapier", "HubSpot", "Zendesk", "Google Sheets"],
        "format": ["CSV", "PDF", "Excel", "JSON", "XML"],
        "os": ["Windows 11", "Windows 10", "macOS Sonoma", "macOS Ventura", "Ubuntu 22.04"],
        "feature": ["export", "import", "dashboard", "reports", "notifications", "search", "sync"],
        "error_msg": ["Connection refused", "Timeout exceeded", "Invalid token", "Access denied", 
                      "Resource not found", "Rate limit exceeded", "Server unavailable"],
        "error_code": ["400", "401", "403", "404", "500", "502", "503", "ETIMEDOUT", "ECONNREFUSED"],
        "endpoint": ["users", "tickets", "reports", "analytics", "settings", "export"],
        "language": ["Spanish", "French", "German", "Chinese", "Japanese", "Portuguese"],
        "competitor": ["Zendesk", "Freshdesk", "Intercom", "ServiceNow", "Jira Service Desk"],
        "compliance": ["GDPR", "HIPAA", "SOC2", "PCI-DSS", "ISO 27001"],
        "promo_code": ["SAVE20", "WELCOME10", "ANNUAL50", "FIRSTMONTH", "UPGRADE25"]
    }
    
    def __init__(self, seed: int = 42):
        """
        Initialize the generator with a random seed for reproducibility.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def _fill_placeholders(self, text: str) -> str:
        """Fill placeholders in text templates with random values."""
        import re
        
        # Find all placeholders like {placeholder_name}
        placeholders = re.findall(r'\{(\w+)\}', text)
        
        for placeholder in placeholders:
            if placeholder in self.PLACEHOLDERS:
                replacement = random.choice(self.PLACEHOLDERS[placeholder])
            elif placeholder == "order_id":
                replacement = str(random.randint(100000, 999999))
            elif placeholder == "invoice_id":
                replacement = f"INV-{random.randint(2024, 2026)}-{random.randint(1000, 9999)}"
            elif placeholder == "amount":
                replacement = str(random.randint(20, 500))
            elif placeholder == "sub_amount":
                replacement = str(random.choice([9.99, 19.99, 29.99, 49.99, 99.99]))
            elif placeholder == "date":
                days_ago = random.randint(1, 30)
                date = datetime.now() - timedelta(days=days_ago)
                replacement = date.strftime("%B %d, %Y")
            elif placeholder == "version":
                replacement = f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
            elif placeholder == "time_ago":
                replacement = random.choice(["yesterday", "2 days ago", "last week", 
                                            "a few hours ago", "this morning"])
            elif placeholder == "size":
                replacement = str(random.choice([10, 50, 100, 250, 500]))
            elif placeholder == "days":
                replacement = str(random.randint(2, 14))
            elif placeholder == "seconds":
                replacement = str(random.randint(10, 60))
            elif placeholder == "memory":
                replacement = str(random.choice([2, 4, 8, 16]))
            elif placeholder == "percent":
                replacement = str(random.choice([50, 75, 90, 95]))
            elif placeholder in ["old_email", "new_email"]:
                replacement = f"user{random.randint(100, 999)}@example.com"
            elif placeholder in ["old_username", "new_username"]:
                replacement = f"user_{random.randint(100, 999)}"
            elif placeholder == "employees":
                replacement = str(random.choice([100, 500, 1000, 5000]))
            elif placeholder == "quantity":
                replacement = str(random.choice([50, 100, 200, 500]))
            else:
                replacement = f"[{placeholder}]"
            
            text = text.replace(f"{{{placeholder}}}", replacement, 1)
        
        return text
    
    def _select_priority(self, category: str) -> str:
        """Select priority based on category-specific weights."""
        weights = self.TEMPLATES[category]["priority_weights"]
        priorities = list(weights.keys())
        probabilities = list(weights.values())
        return random.choices(priorities, weights=probabilities, k=1)[0]
    
    def generate_ticket(self, ticket_id: int = None) -> Dict:
        """
        Generate a single synthetic ticket.
        
        Args:
            ticket_id: Optional ticket ID, auto-generated if not provided
            
        Returns:
            Dictionary containing ticket data
        """
        if ticket_id is None:
            ticket_id = str(uuid.uuid4())[:8].upper()
        
        # Select random category
        category = random.choice(list(self.TEMPLATES.keys()))
        template = self.TEMPLATES[category]
        
        # Generate subject and description
        subject = random.choice(template["subjects"])
        description = random.choice(template["descriptions"])
        
        # Fill placeholders
        subject = self._fill_placeholders(subject)
        description = self._fill_placeholders(description)
        
        # Select priority based on category weights
        priority = self._select_priority(category)
        
        # Generate timestamp
        days_ago = random.randint(0, 90)
        created_at = datetime.now() - timedelta(
            days=days_ago,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        return {
            "ticket_id": f"TKT-{ticket_id}",
            "subject": subject,
            "description": description,
            "category": category,
            "priority": priority,
            "created_at": created_at.isoformat(),
            "status": random.choice(["Open", "In Progress", "Pending", "Resolved", "Closed"])
        }
    
    def generate_dataset(self, n_tickets: int = 500, 
                         balanced: bool = True) -> pd.DataFrame:
        """
        Generate a dataset of synthetic tickets.
        
        Args:
            n_tickets: Number of tickets to generate
            balanced: If True, attempt to balance categories
            
        Returns:
            DataFrame containing generated tickets
        """
        tickets = []
        
        if balanced:
            # Generate roughly equal tickets per category
            categories = list(self.TEMPLATES.keys())
            tickets_per_category = n_tickets // len(categories)
            remainder = n_tickets % len(categories)
            
            for i, category in enumerate(categories):
                count = tickets_per_category + (1 if i < remainder else 0)
                for j in range(count):
                    ticket = self.generate_ticket(len(tickets) + 1)
                    # Override category for balanced distribution
                    ticket["category"] = category
                    ticket["priority"] = self._select_priority(category)
                    tickets.append(ticket)
        else:
            for i in range(n_tickets):
                tickets.append(self.generate_ticket(i + 1))
        
        # Shuffle to mix categories
        random.shuffle(tickets)
        
        return pd.DataFrame(tickets)
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to CSV file."""
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {len(df)} tickets to {filepath}")
    
    def save_to_json(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to JSON file."""
        df.to_json(filepath, orient="records", indent=2, date_format="iso")
        print(f"✅ Saved {len(df)} tickets to {filepath}")


def main():
    """Generate sample dataset."""
    generator = TicketGenerator(seed=42)
    
    # Generate balanced dataset with 500 tickets
    df = generator.generate_dataset(n_tickets=500, balanced=True)
    
    # Display sample
    print("\n📊 Dataset Summary:")
    print(f"   Total tickets: {len(df)}")
    print(f"\n📁 Category Distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\n🎯 Priority Distribution:")
    print(df["priority"].value_counts().to_string())
    print(f"\n📝 Sample Ticket:")
    sample = df.iloc[0]
    print(f"   ID: {sample['ticket_id']}")
    print(f"   Subject: {sample['subject']}")
    print(f"   Category: {sample['category']}")
    print(f"   Priority: {sample['priority']}")
    
    return df


if __name__ == "__main__":
    main()

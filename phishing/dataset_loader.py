"""
dataset_loader.py
-----------------
This module is responsible for creating and loading the email dataset.
It generates a sample dataset of phishing and legitimate emails
that is used for training and testing the model.

Author: Phishing Detection Project
"""

import pandas as pd
import os


def create_sample_dataset():
    """
    Creates a sample dataset of phishing and legitimate emails.
    
    In a real-world project, you would load data from a CSV file
    or a database. Here we generate a representative sample to
    demonstrate the full pipeline.

    Returns:
        pd.DataFrame: A DataFrame with 'text' and 'label' columns.
                      label = 1 means Phishing, label = 0 means Safe/Legitimate
    """

    # -----------------------------------------------------------
    # Phishing email samples (label = 1)
    # -----------------------------------------------------------
    phishing_emails = [
        "Dear user, your account has been compromised. Click here immediately to verify your password and bank details or your account will be suspended.",
        "URGENT: Your PayPal account is limited! Login now at http://paypal-secure-login.xyz to restore access. Provide your credentials.",
        "Congratulations! You have won $1,000,000. Click the link to claim your prize. Verify your bank account number now.",
        "Your Apple ID has been locked. Visit http://apple-id-verify.com and enter your login details urgently.",
        "Security Alert: Suspicious activity detected on your account. Immediately click here to verify your identity and reset your password.",
        "Dear customer, your credit card has been suspended. Update your account information at http://secure-bank-update.net.",
        "You have received a secure message from your bank. Login to http://online-banking-verify.com to read it now.",
        "IMPORTANT: Your email account will be deactivated. Click http://mail-verify-urgent.com to confirm your account details.",
        "Final Warning: Unauthorized access detected. Verify your account at http://account-secure-login.org immediately.",
        "Your Netflix subscription payment failed. Update your billing information now: http://netflix-payment-update.cc",
        "Click here to claim your free iPhone. You have been selected! Verify your address and credit card details.",
        "Action Required: Your bank account shows unusual activity. Call us or click http://bank-alert-secure.net to confirm.",
        "Your Amazon order has a problem. Please login to http://amazon-orders-verify.ru to resolve the issue.",
        "Attention: IRS tax refund available. Provide your social security number and bank info at http://irs-refund-claim.net",
        "Your password expires today. Reset it immediately at http://secure-password-reset.biz or lose access to your account.",
        "URGENT NOTICE: Your account has been flagged for suspicious activity. Verify your credentials to avoid suspension.",
        "Win a $500 gift card! Click here and enter your personal details to claim your reward urgently.",
        "Dear valued member, unusual login detected. Immediately verify your identity via http://member-verify-login.com.",
        "Your subscription is about to expire. Click to renew now and provide your payment details urgently.",
        "Security Warning: Someone tried to access your account. Reset your password now at http://reset-login-secure.xyz.",
        "We noticed an unauthorized transaction on your account. Please verify your identity and banking credentials immediately.",
        "Your Microsoft account requires verification. Click http://microsoft-account-verify.net to confirm your login details.",
        "ALERT: Your email storage is full. Click here to upgrade and enter your card details to continue using the service.",
        "You have 1 new voicemail from the IRS. Call back or click here to listen and verify your tax information.",
        "Your Google account was accessed from a new device. Secure your account now by verifying your password and recovery info.",
        "Confirm your identity to unlock your account. Provide your date of birth, SSN, and credit card number.",
        "Your loan application is approved! Click to access funds. Provide your bank account information to receive money.",
        "FINAL NOTICE: Debt collection proceeding against you. Pay or verify details at http://debt-collect-secure.com.",
        "Urgent: Your insurance policy needs immediate update. Login now to avoid coverage termination.",
        "Click here to verify your shipping address for your pending package. Enter your payment info to release delivery.",
        "Account suspension notice: Your account violates our terms. Login to appeal at http://support-verify-account.net.",
        "You have been pre-approved for a credit card. Click to apply and provide your personal details now.",
        "Dear user, we detected a login from an unknown device. Verify your identity to protect your account immediately.",
        "Your crypto wallet needs re-verification. Click here and provide your wallet key and personal information.",
        "Special offer: Free trial available! Provide credit card details (no charge) at http://free-trial-offer.xyz.",
        "Your DocuSign document is ready. Review and sign at http://docusign-document-verify.com with your credentials.",
        "HR Department: Update your payroll banking information immediately at http://hr-payroll-update.biz.",
        "You have missed a delivery. Click http://delivery-reschedule-now.com and enter your details to reschedule.",
        "Your social media account was hacked. Reset your password immediately by clicking the link below.",
        "Congratulations! Your email was selected in a lottery. Send us your bank details to receive $50,000 prize.",
        "Verify your account to prevent data loss. Click here and enter your username, password, and security answer.",
        "URGENT: Your driving license is suspended. Verify your identity at http://dmv-license-verify.net immediately.",
        "Your cloud storage has been compromised. Login immediately to http://cloud-secure-verify.com to protect your files.",
        "Customer Service: We need to verify your account details. Reply with your password and credit card information.",
        "Your Spotify account will be deleted. Verify your subscription at http://spotify-verify-account.org now.",
        "You have unclaimed funds in your account. Click here to verify your identity and claim $2,500 now.",
        "Security Notice: Your password was recently changed. If this was not you, click here and enter your old password.",
        "Your WhatsApp account is being registered on another device. Enter code 123456 to verify you are the owner.",
        "ALERT: Your phone number linked to suspicious activity. Verify your identity at http://phone-verify-secure.com.",
        "Final opportunity: Claim your tax refund of $1,200. Submit your SSN and bank information before the deadline.",
    ]

    # -----------------------------------------------------------
    # Legitimate / Safe email samples (label = 0)
    # -----------------------------------------------------------
    legitimate_emails = [
        "Hi John, just wanted to follow up on our meeting scheduled for Monday at 10 AM. Let me know if you need to reschedule.",
        "Your monthly bank statement is ready. You can view it by logging into our official banking portal at yourbank.com.",
        "Thank you for your purchase. Your order #12345 has been shipped and will arrive in 3-5 business days.",
        "Team meeting reminder: We have a standup call tomorrow at 9 AM. Please check your calendar for the invite.",
        "Your subscription to the newsletter has been confirmed. Welcome aboard! Expect weekly updates every Monday.",
        "Hi, attached is the report we discussed in yesterday's meeting. Please review and share your feedback.",
        "Your appointment with Dr. Smith is confirmed for June 10 at 2:30 PM. Please arrive 10 minutes early.",
        "Happy Birthday! Wishing you a wonderful day filled with joy and celebration.",
        "Your library book is due for return on June 15. You can renew it online via the library portal.",
        "The package you ordered has been delivered to your front door. Thank you for shopping with us.",
        "Your flight booking confirmation: Flight AA123 on June 20 departing at 6:00 AM from JFK. Enjoy your trip!",
        "Please find attached the invoice for last month's services. Payment is due within 30 days.",
        "Hi Sarah, could you please share the project timeline document when you get a chance? Thanks!",
        "Your tax documents are ready for download in your account portal. Please log in to view them.",
        "We are pleased to inform you that your job application has been received and is under review.",
        "Course enrollment confirmed: Introduction to Python starts on July 1. Check the course portal for materials.",
        "Your gym membership has been renewed for another year. Thank you for being a valued member.",
        "Hi team, the office will be closed on Friday for a public holiday. Enjoy the long weekend!",
        "Your car service appointment is scheduled for Saturday at 11 AM. Please bring your vehicle registration.",
        "The conference call has been moved to 3 PM today. Updated invite has been sent to all participants.",
        "Thanks for attending our webinar! The recording will be available on our website within 48 hours.",
        "Your electricity bill for June is $85.40. It is due by June 25. Pay via our official website.",
        "Hello, we are excited to announce our new product launch happening next week. Stay tuned for updates.",
        "Your visa application status has been updated. Please check the official immigration portal for details.",
        "Meeting notes from yesterday's discussion are attached. Action items are highlighted in yellow.",
        "Your insurance renewal is coming up next month. We will send you a formal notice with updated rates.",
        "Reminder: Please submit your timesheet by end of day Friday to ensure timely payroll processing.",
        "The quarterly results have been published. Please review the attached report for detailed analysis.",
        "Your hotel reservation at Grand Inn is confirmed for check-in on July 5 and check-out on July 8.",
        "Hi, I just wanted to check if you received my last email. Please let me know when you get a chance.",
        "Your blood test results are ready. Please schedule an appointment with your doctor to discuss them.",
        "The project proposal has been approved by the management team. Kickoff meeting is next Tuesday.",
        "Your rental agreement has been renewed for another 12 months starting August 1. Documents are attached.",
        "Happy Thanksgiving! Our offices will be closed November 28-29. We will respond to emails on December 2.",
        "Your warranty for the laptop is valid until December 2026. Keep this email for your records.",
        "We are conducting a survey to improve our services. Your feedback would be greatly appreciated.",
        "Team lunch is scheduled for Friday at noon at the Italian restaurant on Main Street. RSVP by Wednesday.",
        "Your savings account interest has been credited. Check your balance via the official bank app.",
        "The software update has been released. Please check the changelog on our official documentation page.",
        "Thank you for volunteering at the community event. Your contribution made a big difference!",
        "Your annual performance review is scheduled for next week. Please prepare a self-assessment form.",
        "The project deadline has been extended to June 30. Please adjust your timelines accordingly.",
        "Your podcast subscription is active. New episodes drop every Tuesday. Enjoy listening!",
        "Congratulations on your promotion! Your new title is Senior Engineer effective July 1.",
        "The workshop registration is confirmed. Please download the materials from the course website.",
        "Your parcel tracking update: Package has reached the distribution center and will be delivered tomorrow.",
        "Reminder: Board meeting is on Thursday at 4 PM in Conference Room B. Agenda is attached.",
        "Your resume has been shortlisted. We would like to schedule a phone interview at your convenience.",
        "Thank you for your donation to our charity. Your contribution helps us make a difference every day.",
        "Monthly newsletter: Check out our latest blog posts, upcoming events, and product updates inside.",
    ]

    # -----------------------------------------------------------
    # Build the DataFrame
    # -----------------------------------------------------------
    data = {
        "text": phishing_emails + legitimate_emails,
        "label": [1] * len(phishing_emails) + [0] * len(legitimate_emails)
    }

    df = pd.DataFrame(data)

    # Shuffle the dataset so phishing and legitimate emails are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def load_dataset(filepath=None):
    """
    Loads the email dataset.
    
    If a CSV file path is provided and the file exists, it reads from it.
    Otherwise, it falls back to the built-in sample dataset.

    Args:
        filepath (str, optional): Path to a CSV file with 'text' and 'label' columns.

    Returns:
        pd.DataFrame: DataFrame with email text and labels.
    """
    if filepath and os.path.exists(filepath):
        print(f"[INFO] Loading dataset from file: {filepath}")
        df = pd.read_csv(filepath)
        # Make sure required columns exist
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV file must have 'text' and 'label' columns.")
    else:
        print("[INFO] No external dataset found. Using built-in sample dataset.")
        df = create_sample_dataset()

    print(f"[INFO] Dataset loaded: {len(df)} emails total.")
    print(f"       Phishing emails : {df['label'].sum()}")
    print(f"       Legitimate emails: {(df['label'] == 0).sum()}")

    return df


# ---------------------------------------------------------------
# Quick test: run this file directly to preview the dataset
# ---------------------------------------------------------------
if __name__ == "__main__":
    df = load_dataset()
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

"""Email tool for sending emails via SMTP"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool
from ..utils.config import get_config


@tool
def send_mail(
    to: str,
    subject: str,
    body: str,
    html: bool = False
) -> str:
    """Send an email via SMTP.

    Use this tool to send emails to customers, colleagues, or other recipients.
    The email will be sent from the configured sender address.

    Args:
        to: Recipient email address (e.g., "client@example.com")
        subject: Email subject line
        body: Email content (plain text or HTML depending on html parameter)
        html: If True, body is treated as HTML content. Default is False (plain text).

    Returns:
        Success message with recipient or error message

    Examples:
        send_mail("client@example.com", "Votre commande", "Bonjour, votre commande a été expédiée.")
        send_mail("support@velocorp.fr", "Rapport mensuel", "<h1>Rapport</h1><p>Contenu...</p>", html=True)
    """
    try:
        config = get_config()

        # Get email configuration (defaults to Gmail SMTP)
        smtp_server = config.get("smtp_server", "smtp.gmail.com")
        smtp_port = config.get("smtp_port", 465)
        sender_email = config.get("mail_sender", "raphaelcousin.education@gmail.com")
        sender_password = config.get("mail_password")

        if not sender_password:
            return "Error: mail_password not configured. Please add it to your config."

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to

        # Attach body
        if html:
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        # Send email via SSL
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.set_debuglevel(0)  # Set to 1 for debugging
            server.login(sender_email, sender_password)
            refused = server.sendmail(sender_email, to, msg.as_string())

        if refused:
            return f"Email partially failed. Refused recipients: {refused}"
        return f"Email sent successfully to {to} (from {sender_email})"

    except smtplib.SMTPAuthenticationError:
        return "Error: SMTP authentication failed. Check mail_sender and mail_password in config."
    except smtplib.SMTPException as e:
        return f"Error sending email: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

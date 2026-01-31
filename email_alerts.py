"""Email notification system for model monitoring alerts."""

import smtplib
import ssl
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from alerts import Alert, Severity
from config import EmailConfig


@dataclass
class EmailResult:
    """Result of an email send operation."""

    success: bool
    message: str
    recipients_reached: int = 0


def format_alert_html(alert: Alert) -> str:
    """Format a single alert as HTML."""
    severity_color = "#dc3545" if alert.severity == Severity.CRITICAL else "#ffc107"
    severity_bg = "#f8d7da" if alert.severity == Severity.CRITICAL else "#fff3cd"

    return f"""
    <div style="border-left: 4px solid {severity_color};
                background: {severity_bg};
                padding: 12px;
                margin: 10px 0;
                border-radius: 4px;">
        <strong style="color: {severity_color};">
            [{alert.severity.value.upper()}] Day {alert.day}
        </strong>
        <p style="margin: 8px 0 0 0; color: #333;">{alert.message}</p>
    </div>
    """


def build_email_body(alerts: List[Alert]) -> str:
    """Build the complete HTML email body from alerts."""
    critical_count = sum(1 for a in alerts if a.severity == Severity.CRITICAL)
    warning_count = sum(1 for a in alerts if a.severity == Severity.WARNING)

    alerts_html = "\n".join(format_alert_html(alert) for alert in alerts)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        </style>
    </head>
    <body style="padding: 20px; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #333;">Model Performance Alert</h2>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
            <strong>Summary:</strong>
            <span style="color: #dc3545;">{critical_count} Critical</span> |
            <span style="color: #856404;">{warning_count} Warning</span>
        </div>

        {alerts_html}

        <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">

        <p style="color: #666; font-size: 12px;">
            This is an automated alert from the ML Model Monitoring Dashboard.
            Please investigate the flagged metrics to ensure model performance.
        </p>
    </body>
    </html>
    """


def build_email_subject(alerts: List[Alert], prefix: str) -> str:
    """Build the email subject line."""
    critical_count = sum(1 for a in alerts if a.severity == Severity.CRITICAL)

    if critical_count > 0:
        return f"{prefix} CRITICAL: {critical_count} critical alert(s) detected"
    return f"{prefix} WARNING: {len(alerts)} alert(s) detected"


def send_alert_email(
    alerts: List[Alert],
    config: EmailConfig
) -> EmailResult:
    """
    Send email notification for alerts.

    Args:
        alerts: List of Alert objects to include in the email
        config: Email configuration settings

    Returns:
        EmailResult indicating success/failure
    """
    if not config.enabled:
        return EmailResult(
            success=True,
            message="Email alerts disabled in configuration"
        )

    if not alerts:
        return EmailResult(
            success=True,
            message="No alerts to send"
        )

    if not config.recipients:
        return EmailResult(
            success=False,
            message="No recipients configured. Add recipients to EmailConfig."
        )

    if not config.sender_email or not config.sender_password:
        return EmailResult(
            success=False,
            message="Sender email or password not configured. Set sender_email and sender_password."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = build_email_subject(alerts, config.subject_prefix)
    msg["From"] = config.sender_email
    msg["To"] = ", ".join(config.recipients)

    html_body = build_email_body(alerts)
    msg.attach(MIMEText(html_body, "html"))

    try:
        context = ssl.create_default_context()

        if config.use_tls:
            with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
                server.starttls(context=context)
                server.login(config.sender_email, config.sender_password)
                server.sendmail(
                    config.sender_email,
                    config.recipients,
                    msg.as_string()
                )
        else:
            with smtplib.SMTP_SSL(
                config.smtp_host, config.smtp_port, context=context
            ) as server:
                server.login(config.sender_email, config.sender_password)
                server.sendmail(
                    config.sender_email,
                    config.recipients,
                    msg.as_string()
                )

        return EmailResult(
            success=True,
            message=f"Alert email sent successfully",
            recipients_reached=len(config.recipients)
        )

    except smtplib.SMTPAuthenticationError:
        return EmailResult(
            success=False,
            message="SMTP authentication failed. Check sender_email and sender_password."
        )
    except smtplib.SMTPException as e:
        return EmailResult(
            success=False,
            message=f"SMTP error occurred: {str(e)}"
        )
    except OSError as e:
        return EmailResult(
            success=False,
            message=f"Network error: Could not connect to {config.smtp_host}:{config.smtp_port}"
        )


def filter_critical_alerts(alerts: List[Alert]) -> List[Alert]:
    """Filter to only critical severity alerts."""
    return [a for a in alerts if a.severity == Severity.CRITICAL]


def should_send_alert(
    alerts: List[Alert],
    critical_only: bool = False
) -> bool:
    """Determine if an alert email should be sent."""
    if not alerts:
        return False
    if critical_only:
        return any(a.severity == Severity.CRITICAL for a in alerts)
    return True

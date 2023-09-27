import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_txt_email(message_txt, 
                   subject, 
                   to_email="pritthijit.nath.ml@gmail.com"):
  
    # Email configuration
    from_email = "pritthijit.nath.ml@gmail.com"
    
    with open("/homes/pn222/Work/MSc_Project_pn222/dataproc/app.password", "r") as file:
        password = file.read()                
        
    # Create a message object
    message = MIMEMultipart()  
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject
    
    # Email content
    message.attach(MIMEText(message_txt, "plain"))
    
    # Connect to the SMTP server
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_connection = smtplib.SMTP(smtp_server, smtp_port)
    smtp_connection.starttls()
    
    # Login to the email account
    smtp_connection.login(from_email, password)
    
    # Send the email
    smtp_connection.sendmail(from_email, to_email, message.as_string())
    
    # Close the SMTP connection
    smtp_connection.quit()
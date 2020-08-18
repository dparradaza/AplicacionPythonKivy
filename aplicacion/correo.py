import smtplib

def enviar():
    fromaddr = 'user@hotmail.com'
    toaddrs  = 'user@gmail.com'
    msg = 'Ayudame por favor, me encuentro en las ubicación 4.627157, -74.190038!'
    username = 'user@gmail.com'
    password = 'pass'
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()

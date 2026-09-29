from st3215 import ST3215
servo = ST3215("/dev/ttyACM1")
print(servo.ReadPosition(5))

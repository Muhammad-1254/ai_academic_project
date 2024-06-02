import csv
from datetime import datetime

csv_file = 'attendance.csv'

def read_csv():
    data = []
    with open(csv_file, mode='r') as file:
        reader = file.readlines()
        for row in reader:
            print(f"row: {row}")
            data.append(row.split("\n")[0])
    file.close()
    return data


def add_attendance(seat_no:str,student_name:str):
    # Get the current date and time
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the new attendance record
        writer.writerow([seat_no,student_name, date_str, time_str])

    print(f"Attendance added for {student_name} on {date_str} at {time_str}")





def clear_csv():
    with open(csv_file, mode='w') as file:
        pass
    file.close()
    
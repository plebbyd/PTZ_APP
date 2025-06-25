#!/usr/bin/env python3
import os

def fix_opencv_typing():
    typing_file = '/usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py'
    if not os.path.exists(typing_file):
        print(f"File not found: {typing_file}")
        return False
    
    # Read file content
    with open(typing_file, 'r') as f:
        lines = f.readlines()
    
    # Find and comment out the problematic line
    for i, line in enumerate(lines):
        if 'LayerId = cv2.dnn.DictValue' in line:
            lines[i] = '#' + line
            print(f"Found and commented line {i+1}: {line.strip()}")
    
    # Write back the modified content
    with open(typing_file, 'w') as f:
        f.writelines(lines)
    
    print("OpenCV typing file updated successfully")
    return True

if __name__ == '__main__':
    fix_opencv_typing()
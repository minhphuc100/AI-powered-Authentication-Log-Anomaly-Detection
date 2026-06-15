import pandas as pd
import re
import os

def extract_field_from_message(message, pattern):
    """
    Hàm phụ trợ để trích xuất dữ liệu từ chuỗi văn bản Message bằng Regex.
    """
    if pd.isna(message):
        return None
    match = re.search(pattern, str(message))
    if match:
        return match.group(1).strip()
    return None

def parse_windows_auth_logs(input_filepath, output_filepath):
    """
    Hàm chính để đọc file log thô, bóc tách đặc trưng và lưu ra file dữ liệu sạch.
    """
    print(f"Đang đọc dữ liệu thô từ: {input_filepath}")
    
    try:
        # Đọc file CSV, thường log của Windows xuất ra sẽ có cột 'Id', 'TimeCreated', 'Message'
        df = pd.read_csv(input_filepath, encoding='utf-8')
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['Id', 'TimeCreated', 'Message']
    for col in required_columns:
        if col not in df.columns:
            print(f"Lỗi: Không tìm thấy cột '{col}' trong dữ liệu gốc.")
            return None

    print(f"Số lượng log ban đầu: {len(df)} dòng. Bắt đầu phân tích cú pháp...")

    # 1. Trích xuất Tên tài khoản (Account Name)
    # Lấy Account Name thứ 2 (thường thuộc phần New Logon hoặc Target Account) để tránh lấy tên SYSTEM của máy
    df['AccountName'] = df['Message'].apply(
        lambda x: extract_field_from_message(x, r"Account Name:\s+(.*?)(?:\r|\n|$)")
    )

    # 2. Trích xuất Loại đăng nhập (Logon Type)
    df['LogonType'] = df['Message'].apply(
        lambda x: extract_field_from_message(x, r"Logon Type:\s+(\d+)")
    )

    # 3. Trích xuất Địa chỉ IP nguồn (Source Network Address / Source IP)
    # Rất quan trọng cho việc đếm unique_src_ip_1h sau này
    df['SourceIP'] = df['Message'].apply(
        lambda x: extract_field_from_message(x, r"Source Network Address:\s+([^\r\n]+)")
    )
    # Lọc bỏ các IP rỗng hoặc giá trị '-'
    df['SourceIP'] = df['SourceIP'].replace('-', 'Unknown')

    # 4. Gán nhãn (Labeling) trực tiếp dựa trên Event ID
    # Event ID 4624 = Đăng nhập thành công (Normal)
    # Event ID 4625 = Đăng nhập thất bại (Anomaly/Brute Force)
    df['Label'] = df['Id'].apply(lambda x: 'Anomaly' if str(x) == '4625' else 'Normal')

    # 5. Chuyển đổi định dạng thời gian cho chuẩn xác
    df['TimeCreated'] = pd.to_datetime(df['TimeCreated'], errors='coerce')

    # 6. Lọc lại các cột cần thiết để làm dữ liệu sạch
    clean_df = df[['TimeCreated', 'Id', 'AccountName', 'LogonType', 'SourceIP', 'Label']]
    
    # Sắp xếp lại log theo đúng trình tự thời gian
    clean_df = clean_df.sort_values(by='TimeCreated').reset_index(drop=True)

    # Lưu ra file CSV mới
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    clean_df.to_csv(output_filepath, index=False, encoding='utf-8')
    
    print(f"Hoàn thành! Đã lưu {len(clean_df)} dòng dữ liệu sạch tại: {output_filepath}")
    return clean_df

if __name__ == "__main__":
    # Đường dẫn file theo đúng cấu trúc thư mục của nhóm bạn
    INPUT_FILE = r"data\raw\windows_auth_logs.csv"
    OUTPUT_FILE = r"data\processed\parsed_auth_logs.csv"
    
    # Chạy pipeline
    parse_windows_auth_logs(INPUT_FILE, OUTPUT_FILE)
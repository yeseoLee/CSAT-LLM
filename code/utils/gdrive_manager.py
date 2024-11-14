import io
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from loguru import logger
import pandas as pd

from .util import load_env_file


SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


class GoogleDriveManager:
    def __init__(self):
        load_env_file("../config/.env")
        self.service = self.get_drive_service()
        self.root_folder_id = os.getenv("GDRIVE_FOLDER_ID")
        self.credentials = os.getenv("GDRIVE_CREDENTIALS")
        self.token = os.getenv("GDRIVE_TOKEN")

    def get_drive_service(self):
        creds = None
        if os.path.exists(self.token):
            creds = Credentials.from_authorized_user_file(self.token, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.token, "w") as token:
                token.write(creds.to_json())

        return build("drive", "v3", credentials=creds)

    def find_folder_id_by_name(self, folder_name, parent_folder_id=None):
        """폴더명으로 폴더 ID 찾기"""
        if not parent_folder_id:
            parent_folder_id = self.root_folder_id

        # 특정 폴더명과 정확히 일치하는 폴더 검색 쿼리
        query = f"name='{folder_name}' and "
        query += f"'{parent_folder_id}' in parents and "
        query += "mimeType='application/vnd.google-apps.folder' and "
        query += "trashed=false"

        try:
            results = (
                self.service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="files(id, name)",
                    pageSize=1,  # 첫 번째 일치하는 폴더만 필요
                )
                .execute()
            )

            files = results.get("files", [])

            if not files:
                logger.info(f"폴더를 찾을 수 없습니다: {folder_name}")
                return None

            return files[0]["id"]

        except Exception as e:
            logger.info(f"폴더 검색 중 오류 발생: {str(e)}")
            return None

    def upload_json_data(self, json_string, filename, folder_id=None):
        """직렬화된 JSON string 직접 업로드"""
        try:
            # 메모리 스트림으로 변환
            file_stream = io.BytesIO(json_string.encode("utf-8"))

            # 파일 메타데이터 설정
            file_metadata = {"name": filename, "mimeType": "application/json"}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            # 미디어 객체 생성
            media = MediaIoBaseUpload(file_stream, mimetype="application/json", resumable=True)

            # 파일 업로드
            file = self.service.files().create(body=file_metadata, media_body=media, fields="id, name").execute()
            return file

        except Exception as e:
            logger.info(f"Error uploading JSON data: {str(e)}")
            return None

    def upload_dataframe(self, dataframe, filename, folder_id=None):
        """Pandas DataFrame 직접 업로드"""
        try:
            # DataFrame을 CSV 스트림으로 변환
            buffer = io.StringIO()
            dataframe.to_csv(buffer, index=False)
            file_stream = io.BytesIO(buffer.getvalue().encode("utf-8"))

            # 파일 메타데이터 설정
            file_metadata = {"name": filename, "mimeType": "text/csv"}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            # 미디어 객체 생성
            media = MediaIoBaseUpload(file_stream, mimetype="text/csv", resumable=True)

            # 파일 업로드
            file = self.service.files().create(body=file_metadata, media_body=media, fields="id, name").execute()
            return file

        except Exception as e:
            logger.info(f"Error uploading DataFrame: {str(e)}")
            return None

    def upload_output_csv(self, user_name, filepath):
        df = pd.read_csv(filepath)
        basename = os.path.basename(filepath)

        # 실험자명으로 폴더명 찾기
        folder_id = self.find_folder_id_by_name(user_name)
        _ = self.upload_dataframe(df, basename, folder_id)

        gdrive_url = os.path.join("https://drive.google.com/drive/folders", folder_id)
        logger.info(f"구글 드라이브에 업로드 되었습니다: {gdrive_url}")


if __name__ == "__main__":
    drive_manager = GoogleDriveManager()
    # 파일 목록 조회
    files = drive_manager.list_folder_files()
    for file in files:
        logger.info(f"Name: {file['name']}, ID: {file['id']}")

import io
import json
import os.path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from loguru import logger
import pandas as pd


SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


class GoogleDriveManager:
    def __init__(self):
        config_folder = os.path.join(os.path.dirname(__file__), "..", "..", "config")
        load_dotenv(os.path.join(config_folder, ".env"))
        self.config_folder = config_folder
        self.root_folder_id = os.getenv("GDRIVE_FOLDER_ID")
        self.credentials = os.getenv("GDRIVE_CREDENTIALS")
        self.token = os.getenv("GDRIVE_TOKEN")
        self.is_create_token = os.getenv("GDRIVE_CREATE_TOKEN")

        # 환경 변수 검증 추가
        if not all([self.root_folder_id, self.credentials, self.token]):
            logger.error(f"필수 환경 변수가 설정되지 않았습니다. {[self.root_folder_id, self.credentials, self.token]}")
            raise ValueError("필수 환경 변수가 설정되지 않았습니다.")

        self.service = self.get_drive_service()

    def get_drive_service(self):
        creds = None
        if os.path.exists(self.token):
            creds = Credentials.from_authorized_user_file(self.token, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif self.is_create_token == "true":
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials, scopes=SCOPES)
                creds = flow.run_local_server(port=0, open_browser=False)
                # 리프레시 토큰 저장
                with open(self.token, "w") as token_file:
                    token_data = {
                        "refresh_token": creds.refresh_token,
                        "token": creds.token,
                        "token_uri": creds.token_uri,
                        "client_id": creds.client_id,
                        "client_secret": creds.client_secret,
                        "scopes": creds.scopes,
                    }
                    json.dump(token_data, token_file)
                    logger.info("구글 드라이브 토큰이 갱신되었습니다.")
            else:
                logger.error("구글 드라이브 업로드 실패. 토큰을 갱신을 요청하세요.")
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

    def list_folder_files(self, folder_id=None):
        """폴더 내 파일 목록 조회"""
        if not folder_id:
            folder_id = self.root_folder_id
        query = f"'{folder_id}' in parents and trashed=false"

        try:
            results = (
                self.service.files()
                .list(
                    q=query,
                    pageSize=100,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                )
                .execute()
            )

            return results.get("files", [])
        except Exception as e:
            logger.info(f"Error listing files: {str(e)}")
            return []

    def upload_yaml_file(self, file_path, filename, folder_id=None):
        """YAML 파일 경로를 받아서 업로드"""
        try:
            # 파일 메타데이터 설정
            file_metadata = {"name": filename, "mimeType": "application/x-yaml"}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            # 미디어 객체 생성
            media = MediaFileUpload(file_path, mimetype="application/x-yaml", resumable=True)

            # 파일 업로드
            file = self.service.files().create(body=file_metadata, media_body=media, fields="id, name").execute()

            logger.debug(f"Successfully uploaded {filename} to Google Drive")
            return file

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error uploading YAML file: {str(e)}")
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
            logger.error(f"Error uploading DataFrame: {str(e)}")
            return None

    def upload_exp(self, user_name, output_path, config_path=None):
        df = pd.read_csv(output_path)
        df_basename = os.path.basename(output_path)

        if config_path is None:
            config_path = os.path.join(self.config_folder, "config.yaml")
        config_basename = df_basename.replace("output.csv", "config.yaml")

        # 실험자명으로 폴더명 찾기
        folder_id = self.find_folder_id_by_name(user_name)
        _ = self.upload_dataframe(df, df_basename, folder_id)
        _ = self.upload_yaml_file(config_path, config_basename, folder_id)

        gdrive_url = os.path.join("https://drive.google.com/drive/folders", folder_id)
        logger.info(f"구글 드라이브에 업로드 되었습니다: {gdrive_url}")


if __name__ == "__main__":
    os.chdir("..")
    load_dotenv("../config/.env")
    drive_manager = GoogleDriveManager()
    # 파일 목록 조회
    files = drive_manager.list_folder_files()
    for file in files:
        logger.info(f"Name: {file['name']}, ID: {file['id']}")

import sqlite3
from datetime import datetime
from typing import List

from pydantic import BaseModel

from core.repositories.repo_abstract import AbstractRepository


class FileItem(BaseModel):
    file_name: str
    file_name_orig: str
    user_id: int
    created_at: datetime
    processing_status: str = ""
    vector_store_id: str = ""


class FilesRepository(AbstractRepository):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_db()

    def _init_db(self):
        with self._get_db_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_files (
                file_name TEXT PRIMARY KEY,
                file_name_orig TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                processing_status TEXT DEFAULT '',
                vector_store_id TEXT DEFAULT ''
            )
            """)

            # Add vector_store_id column if it doesn't exist
            try:
                conn.execute("ALTER TABLE user_files ADD COLUMN vector_store_id TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass

            conn.commit()

    def create_file_sync(self, file: FileItem) -> bool:
        with self._get_db_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO user_files 
                    (file_name, file_name_orig, user_id, created_at, processing_status, vector_store_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file.file_name,
                        file.file_name_orig,
                        file.user_id,
                        file.created_at.isoformat(),
                        file.processing_status,
                        file.vector_store_id
                    )
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_files_by_filter_sync(self, filter: str, params: tuple = ()) -> List[FileItem]:
        """
        Get files based on a custom filter expression.

        Args:
            filter: SQL WHERE clause (without the 'WHERE' keyword)
            params: Parameters to be used with the filter expression

        Returns:
            List of FileItem objects matching the filter
        """
        with self._get_db_connection() as conn:
            query = f"""
            SELECT file_name, file_name_orig, user_id, created_at, processing_status, vector_store_id
            FROM user_files
            WHERE {filter}
            ORDER BY created_at DESC
            """
            cursor = conn.execute(query, params)

            files = []
            for row in cursor.fetchall():
                files.append(FileItem(
                    file_name=row[0],
                    file_name_orig=row[1],
                    user_id=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    processing_status=row[4],
                    vector_store_id=row[5]
                ))

            return files

    def delete_file_sync(self, file_name: str) -> bool:
        with self._get_db_connection() as conn:
            try:
                cursor = conn.execute(
                    "DELETE FROM user_files WHERE file_name = ?",
                    (file_name,)
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error:
                return False

    def get_all_files_sync(self) -> List[FileItem]:
        """
        Get all files in the repository.

        Returns:
            List of all FileItem objects ordered by creation date (newest first)
        """
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT file_name, file_name_orig, user_id, created_at, processing_status, vector_store_id
                FROM user_files
                ORDER BY created_at DESC
                """
            )

            files = []
            for row in cursor.fetchall():
                files.append(FileItem(
                    file_name=row[0],
                    file_name_orig=row[1],
                    user_id=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    processing_status=row[4],
                    vector_store_id=row[5]
                ))

            return files

    def update_file_sync(self, file_name: str, file_item: FileItem) -> bool:
        """
        Update file information in the repository.

        Args:
            file_name: The file name that serves as the primary key
            file_item: FileItem object containing the updated information

        Returns:
            True if the update was successful, False otherwise
        """
        with self._get_db_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    UPDATE user_files 
                    SET file_name_orig = ?, user_id = ?, created_at = ?, processing_status = ?, vector_store_id = ?
                    WHERE file_name = ?
                    """,
                    (
                        file_item.file_name_orig,
                        file_item.user_id,
                        file_item.created_at.isoformat(),
                        file_item.processing_status,
                        file_item.vector_store_id,
                        file_name
                    )
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error:
                return False

    async def create_file(self, file: FileItem) -> bool:
        return await self._run_in_thread(self.create_file_sync, file)

    async def get_files_by_filter(self, filter: str, params: tuple = ()) -> List[FileItem]:
        """
        Async version of get_files_by_filter_sync.

        Args:
            filter: SQL WHERE clause (without the 'WHERE' keyword)
            params: Parameters to be used with the filter expression

        Returns:
            List of FileItem objects matching the filter

        Examples:
            Get all PDF files:
            files = await repo.get_files_by_filter("file_ext = ?", ("pdf",))

            Get files with a specific processing status for a user:
            files = await repo.get_files_by_filter("user_id = ? AND processing_status = ?", (user_id, "completed"))
        """
        return await self._run_in_thread(self.get_files_by_filter_sync, filter, params)

    async def delete_file(self, file_name: str) -> bool:
        return await self._run_in_thread(self.delete_file_sync, file_name)

    async def get_all_files(self) -> List[FileItem]:
        """
        Async version of get_all_files_sync.

        Returns:
            List of all FileItem objects ordered by creation date (newest first)
        """
        return await self._run_in_thread(self.get_all_files_sync)


    async def update_file(self, file_name: str, file_item: FileItem) -> bool:
        """
        Async version of update_file_sync.

        Args:
            file_name: The file name that serves as the primary key
            file_item: FileItem object containing the updated information

        Returns:
            True if the update was successful, False otherwise

        Examples:
            Update file information:
            updated_file = FileItem(
                file_name="document.pdf",
                file_name_orig="new_name.pdf",
                user_id=123,
                created_at=datetime.now(),
                processing_status="completed",
                vector_store_id="vs_123456"
            )
            success = await repo.update_file("document.pdf", updated_file)
        """
        return await self._run_in_thread(self.update_file_sync, file_name, file_item)

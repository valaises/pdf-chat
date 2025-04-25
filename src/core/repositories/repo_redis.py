import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import redis
import numpy as np

from core.logger import info, error, warn, exception


@dataclass
class VectorItem:
    """Dataclass representing a vector item to be stored in Redis."""
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Dataclass representing a search result from Redis vector search."""
    id: str
    score: float
    vector: Optional[bytes] = None
    metadata: Dict[str, Any] = None

# todo: drop redis indices of deleted documents, execute in w_cleanup
class RedisRepository:
    def __init__(
            self,
            host='redis',
            port=6379
    ):
        self.host = host
        self.port = port
        self.redis = None

    def trigger_save(self) -> bool:
        """
        Trigger an immediate save of the dataset to disk.

        This method forces Redis to create an RDB snapshot immediately,
        ensuring all current data is persisted to disk.

        Returns:
            bool: True if save was triggered successfully, False otherwise
        """
        try:
            # Execute SAVE command (blocking) or BGSAVE (non-blocking)
            # Using BGSAVE to avoid blocking the Redis server
            result = self.redis.bgsave()
            info(f"Redis save triggered: {result}")
            return True
        except Exception as e:
            error(f"Error triggering Redis save: {e}")
            return False

    def connect(self):
        """Establish a connection to the Redis server."""
        self.redis = redis.Redis(host=self.host, port=self.port)
        info("Connected to Redis!")

    def close(self):
        """Close the connection to the Redis server."""
        if self.redis:
            self.redis.close()
            info("Connection to Redis closed.")

    def index_exists(self, index_name: str) -> bool:
        """
        Check if a vector index exists in Redis.

        Args:
            index_name: Name of the index to check

        Returns:
            bool: True if the index exists, False otherwise
        """
        if not self.redis:
            self.connect()

        try:
            self.redis.execute_command("FT.INFO", index_name)
            return True
        except Exception as e:
            if "Unknown index name" in str(e):
                return False
            # Re-raise other exceptions
            raise e

    def create_vector_index(
            self, index_name: str, dimension: int, distance_metric: str = "COSINE",
            prefix: str = None, index_type: str = "HASH"
    ) -> bool:
        """
        Create a vector index in Redis.

        Args:
            index_name: Name of the index
            dimension: Dimension of vectors to be stored
            distance_metric: Distance metric for similarity search (COSINE, L2, IP)
            prefix: Key prefix to index (defaults to index_name:)
            index_type: Type of index (HASH or JSON)

        Returns:
            bool: True if index was created successfully or already exists, False otherwise
        """
        try:
            # Set default prefix if not provided
            if prefix is None:
                prefix = f"{index_name}:"

            # Configure vector parameters
            vector_args = [
                "TYPE", "FLOAT32",
                "DIM", str(dimension),
                "DISTANCE_METRIC", distance_metric
            ]

            # Build the command based on index type
            if index_type.upper() == "JSON":
                create_cmd = [
                    "FT.CREATE", index_name,
                    "ON", "JSON",
                    "PREFIX", "1", prefix,
                    "SCHEMA",
                    "$.vector", "AS", "vector", "VECTOR", "FLAT", "6"
                ]
                create_cmd.extend(vector_args)
            else:  # Default HASH type
                create_cmd = [
                    "FT.CREATE", index_name,
                    "ON", "HASH",
                    "PREFIX", "1", prefix,
                    "SCHEMA",
                    "vector", "VECTOR", "FLAT", "6"
                ]
                create_cmd.extend(vector_args)

            self.redis.execute_command(*create_cmd)
            info(f"Vector index '{index_name}' created successfully with {dimension} dimensions")
            return True

        except Exception as e:
            if "Index already exists" in str(e):
                warn(f"Index '{index_name}' already exists")
                return True
            exception(f"Error creating vector index: {e}")
            return False

    def add_vectors(self, index_name: str, vectors: List[VectorItem]) -> None:
        """
        Add multiple vectors to the Redis vector index.

        Args:
            index_name: Name of the index
            vectors: List of tuples containing (id, vector, metadata)
                     where metadata is a dictionary of additional fields
        """
        pipeline = self.redis.pipeline()

        for vec in vectors:
            key = f"{index_name}:{vec.id}"

            # Prepare data for storage
            data = {
                "vector": np.array(vec.vector).astype(np.float32).tobytes(),
                "id": vec.id
            }

            # Add metadata fields
            if vec.metadata:
                for k, v in vec.metadata.items():
                    if isinstance(v, (dict, list)):
                        data[k] = json.dumps(v)
                    else:
                        data[k] = v

            # Add to pipeline
            # WARNING: in case of paragraphs with same text, they will have same id, one of them will be overwritten
            pipeline.hset(key, mapping=data)

        # Execute pipeline
        pipeline.execute()
        info(f"Added {len(vectors)} vectors to index '{index_name}'")

    def get_all_vector_ids(self, index_name: str) -> List[str]:
        """
        Retrieve IDs of all vectors stored in the specified index.

        This method scans the Redis database for all keys matching the index pattern
        and extracts their IDs.

        Args:
            index_name: Name of the index to retrieve vector IDs from

        Returns:
            List[str]: A list of all vector IDs in the specified index

        Note:
            - This operation may be expensive for large datasets as it scans all keys
            - The returned IDs are sorted alphabetically
        """
        try:
            # Pattern to match all keys in the index
            pattern = f"{index_name}:*"

            # Use SCAN to iterate through keys matching the pattern
            cursor = 0
            all_keys = []

            while True:
                cursor, keys = self.redis.scan(cursor=cursor, match=pattern)
                all_keys.extend(keys)

                # Break when we've scanned all keys
                if cursor == 0:
                    break

            # Extract IDs from keys (remove the prefix)
            vector_ids = [key.decode('utf-8').split(':', 1)[1] for key in all_keys]

            info(f"Retrieved {len(vector_ids)} vector IDs from index '{index_name}'")
            return vector_ids

        except Exception as e:
            exception(f"Error retrieving vector IDs from index '{index_name}': {e}")
            return []

    def search_vectors(
            self,
            index_name: str,
            query_vector: List[float],
            top_k: int = 10,
            filter_expr: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors in the Redis vector index using KNN.

        This method performs a K-nearest neighbors search in the specified Redis vector index
        to find vectors similar to the provided query vector. The search can be further refined
        with an optional filter expression.

        Args:
            index_name: Name of the Redis vector index to search in
            query_vector: The query vector as a list of float values
            top_k: Maximum number of results to return (default: 10)
            filter_expr: Optional Redis query filter expression to narrow down results
                         (e.g., "@category:{electronics}")

        Returns:
            List[SearchResult]: A list of SearchResult objects containing the matched vectors,
                               sorted by similarity score. Each result includes the document ID,
                               similarity score, vector data (if available), and metadata.

        Note:
            - The similarity score is determined by the distance metric configured for the index
              (COSINE, L2, IP)
            - Lower scores typically indicate higher similarity, depending on the distance metric
            - Returns an empty list if an error occurs during the search
        """
        if not self.redis:
            self.connect()

        # Convert query vector to bytes
        query_vec_bytes = np.array(query_vector).astype(np.float32).tobytes()

        try:
            # Construct the query string for vector search
            query = f"*=>[KNN {top_k} @vector $query_vec AS score]"
            if filter_expr:
                query = f"{filter_expr}=>[KNN {top_k} @vector $query_vec AS score]"

            # Execute the search
            result = self.redis.execute_command(
                "FT.SEARCH", index_name, query,
                "PARAMS", 2, "query_vec", query_vec_bytes,
                "SORTBY", "score", "ASC",
                "DIALECT", 2
            )

            return self._process_search_results(result)

        except Exception as e:
            exception(f"Error searching vectors: {e}")
            return []

    def _process_search_results(self, result) -> List[SearchResult]:
        """
        Helper method to process Redis search results into SearchResult objects.

        This method parses the raw Redis FT.SEARCH response format and converts it into
        a list of structured SearchResult objects. The Redis response follows this pattern:
        [total_count, key1, [field1, value1, field2, value2, ...], key2, [...], ...]

        Args:
            result: Raw response from Redis FT.SEARCH command, containing the count of results
                   followed by alternating document keys and field arrays

        Returns:
            List[SearchResult]: A list of SearchResult objects containing the parsed data,
                               with each object having id, score, vector (if available),
                               and metadata fields extracted from the Redis response

        Note:
            - The first element in the result is skipped as it contains the count of results
            - Each document's data is stored at alternating positions (hence the step of 2)
            - The 'vector' and 'score' fields are extracted from metadata and set as
              dedicated properties on the SearchResult object
        """
        if not result or len(result) <= 1:
            return []

        processed_results = []

        # Skip the first element (count of results)
        for i in range(1, len(result), 2):
            if i + 1 >= len(result):
                continue

            key = result[i].decode('utf-8')
            doc_id = key.split(':')[-1]

            # Get the fields array
            fields_array = result[i + 1]

            # Process fields
            fields = {
                fields_array[j].decode('utf-8'): self._process_field_value(fields_array[j + 1])
                for j in range(0, len(fields_array), 2)
                if j + 1 < len(fields_array)
            }

            # Extract special fields
            vector = fields.pop('vector', None)
            score = fields.pop('score', 0.0)

            # Create SearchResult instance
            processed_results.append(SearchResult(
                id=doc_id,
                score=score,
                vector=vector,
                metadata=fields
            ))

        return processed_results

    def _process_field_value(self, value) -> Any:
        """
        Process a field value from Redis response into an appropriate Python type.

        This method handles the conversion of Redis response values:
        - If the value is not bytes, it's returned as-is
        - If the value is bytes, it attempts to decode it as UTF-8
        - If the decoded string looks like JSON (starts with '{' or '['), it attempts to parse it
        - If JSON parsing fails or the string doesn't look like JSON, returns the decoded string
        - If UTF-8 decoding fails, returns the raw bytes (likely binary data)

        Args:
            value: The value from Redis response to process

        Returns:
            The processed value as an appropriate Python type (dict, list, str, or bytes)
        """
        if not isinstance(value, bytes):
            return value

        try:
            # Try to decode as string
            decoded = value.decode('utf-8')

            # Try to parse as JSON if it looks like JSON
            if decoded.startswith('{') or decoded.startswith('['):
                try:
                    return json.loads(decoded)
                except json.JSONDecodeError:
                    pass
            return decoded
        except UnicodeDecodeError:
            # If it can't be decoded as UTF-8, return the raw bytes
            return value

"""
SQLite Database Operations for Ticket Storage

Provides database schema creation, data ingestion from CSV/JSON,
and query operations for the ticket classification system.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd


class TicketDatabase:
    """
    SQLite database manager for support tickets.
    
    Provides methods for:
        - Creating database schema
        - Ingesting tickets from CSV/JSON
        - Querying and retrieving ticket data
        - Basic CRUD operations
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id TEXT UNIQUE NOT NULL,
        subject TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL,
        priority TEXT NOT NULL,
        created_at TEXT,
        status TEXT DEFAULT 'Open',
        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
    CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);
    CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
    CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at);
    """
    
    def __init__(self, db_path: str = "data/tickets.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
    
    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def create_schema(self) -> None:
        """Create database tables and indexes."""
        cursor = self.connection.cursor()
        cursor.executescript(self.SCHEMA)
        self.connection.commit()
        print(f"✅ Database schema created at {self.db_path}")
    
    def insert_ticket(self, ticket: Dict) -> int:
        """
        Insert a single ticket into the database.
        
        Args:
            ticket: Dictionary with ticket data
            
        Returns:
            Row ID of inserted ticket
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tickets 
            (ticket_id, subject, description, category, priority, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket.get("ticket_id"),
            ticket.get("subject"),
            ticket.get("description"),
            ticket.get("category"),
            ticket.get("priority"),
            ticket.get("created_at"),
            ticket.get("status", "Open")
        ))
        self.connection.commit()
        return cursor.lastrowid
    
    def insert_many(self, tickets: List[Dict]) -> int:
        """
        Bulk insert multiple tickets.
        
        Args:
            tickets: List of ticket dictionaries
            
        Returns:
            Number of tickets inserted
        """
        cursor = self.connection.cursor()
        data = [
            (
                t.get("ticket_id"),
                t.get("subject"),
                t.get("description"),
                t.get("category"),
                t.get("priority"),
                t.get("created_at"),
                t.get("status", "Open")
            )
            for t in tickets
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO tickets 
            (ticket_id, subject, description, category, priority, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)
        self.connection.commit()
        return len(data)
    
    def load_from_csv(self, csv_path: str) -> int:
        """
        Load tickets from CSV file into database.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Number of tickets loaded
        """
        df = pd.read_csv(csv_path)
        tickets = df.to_dict(orient="records")
        count = self.insert_many(tickets)
        print(f"✅ Loaded {count} tickets from {csv_path}")
        return count
    
    def load_from_json(self, json_path: str) -> int:
        """
        Load tickets from JSON file into database.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Number of tickets loaded
        """
        df = pd.read_json(json_path)
        tickets = df.to_dict(orient="records")
        count = self.insert_many(tickets)
        print(f"✅ Loaded {count} tickets from {json_path}")
        return count
    
    def get_all_tickets(self) -> pd.DataFrame:
        """Retrieve all tickets as DataFrame."""
        return pd.read_sql_query("SELECT * FROM tickets", self.connection)
    
    def get_ticket_by_id(self, ticket_id: str) -> Optional[Dict]:
        """
        Get a specific ticket by its ID.
        
        Args:
            ticket_id: Ticket ID to retrieve
            
        Returns:
            Ticket dictionary or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_tickets_by_category(self, category: str) -> pd.DataFrame:
        """Get all tickets in a specific category."""
        return pd.read_sql_query(
            "SELECT * FROM tickets WHERE category = ?",
            self.connection,
            params=(category,)
        )
    
    def get_tickets_by_priority(self, priority: str) -> pd.DataFrame:
        """Get all tickets with a specific priority."""
        return pd.read_sql_query(
            "SELECT * FROM tickets WHERE priority = ?",
            self.connection,
            params=(priority,)
        )
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get count of tickets per category."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM tickets 
            GROUP BY category 
            ORDER BY count DESC
        """)
        return {row["category"]: row["count"] for row in cursor.fetchall()}
    
    def get_priority_counts(self) -> Dict[str, int]:
        """Get count of tickets per priority."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT priority, COUNT(*) as count 
            FROM tickets 
            GROUP BY priority 
            ORDER BY count DESC
        """)
        return {row["priority"]: row["count"] for row in cursor.fetchall()}
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM tickets")
        total = cursor.fetchone()[0]
        
        return {
            "total_tickets": total,
            "category_distribution": self.get_category_counts(),
            "priority_distribution": self.get_priority_counts()
        }
    
    def search_tickets(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search tickets by subject or description.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            DataFrame of matching tickets
        """
        return pd.read_sql_query(
            """
            SELECT * FROM tickets 
            WHERE subject LIKE ? OR description LIKE ?
            LIMIT ?
            """,
            self.connection,
            params=(f"%{query}%", f"%{query}%", limit)
        )
    
    def clear_all(self) -> None:
        """Delete all tickets from database."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM tickets")
        self.connection.commit()
        print("⚠️ All tickets cleared from database")


def main():
    """Demo database operations."""
    from generator import TicketGenerator
    
    # Generate sample data
    generator = TicketGenerator(seed=42)
    df = generator.generate_dataset(n_tickets=100)
    
    # Initialize database
    with TicketDatabase("data/tickets.db") as db:
        db.create_schema()
        
        # Insert tickets
        tickets = df.to_dict(orient="records")
        count = db.insert_many(tickets)
        print(f"\n📊 Inserted {count} tickets")
        
        # Show statistics
        stats = db.get_statistics()
        print(f"\n📈 Database Statistics:")
        print(f"   Total: {stats['total_tickets']}")
        print(f"   Categories: {stats['category_distribution']}")
        print(f"   Priorities: {stats['priority_distribution']}")


if __name__ == "__main__":
    main()

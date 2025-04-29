# Database Scripts

This directory contains scripts related to the NeuraShield database setup and management.

## Scripts

- `init-db.sql` - SQL initialization script for setting up the NeuraShield database

## Usage

The SQL script can be used to initialize a new MySQL or MariaDB database for NeuraShield:

```bash
# Example: Initialize the database
mysql -u root -p < scripts/database/init-db.sql
```

## Notes

- The initialization script creates the necessary tables with proper indexes
- A default admin user is created with username `admin` and password `admin123`
- Default settings are configured for blockchain integration and IPFS storage
- Make sure to change the default credentials in a production environment

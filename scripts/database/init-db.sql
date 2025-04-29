-- NeuraShield Database Initialization Script

-- Create the neurashield database if it doesn't exist
CREATE DATABASE IF NOT EXISTS neurashield;

-- Switch to the neurashield database
USE neurashield;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create events table for local caching of blockchain events
CREATE TABLE IF NOT EXISTS events (
    id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    type VARCHAR(100) NOT NULL,
    details TEXT NOT NULL,
    ipfs_hash VARCHAR(100),
    blockchain_tx_id VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create settings table
CREATE TABLE IF NOT EXISTS settings (
    id VARCHAR(36) PRIMARY KEY,
    key_name VARCHAR(100) NOT NULL UNIQUE,
    value TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36),
    action VARCHAR(100) NOT NULL,
    details TEXT,
    ip_address VARCHAR(45),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Insert default admin user (password: admin123)
INSERT INTO users (id, username, password_hash, role, email)
VALUES (
    'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
    'admin',
    '$2a$10$jQvqGpkzGRFvMVhgj6tYGO6YrXZjVNJ6nbFyXHwDGwuVlZWRR7ZFi',
    'admin',
    'admin@neurashield.io'
) ON DUPLICATE KEY UPDATE id=id;

-- Insert default settings
INSERT INTO settings (id, key_name, value, description)
VALUES 
    ('s1e2t3t4-i5n6g7s8-9a0b-cdef-1234567890a1', 'blockchain_enabled', 'true', 'Enable blockchain integration'),
    ('s1e2t3t4-i5n6g7s8-9a0b-cdef-1234567890a2', 'ipfs_enabled', 'true', 'Enable IPFS for large event data storage'),
    ('s1e2t3t4-i5n6g7s8-9a0b-cdef-1234567890a3', 'ai_model_version', 'latest', 'The AI model version to use')
ON DUPLICATE KEY UPDATE id=id;

-- Create indexes
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_type ON events(type);
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);

-- Grant privileges
GRANT ALL PRIVILEGES ON neurashield.* TO 'neurashield_user'@'%';
FLUSH PRIVILEGES; 
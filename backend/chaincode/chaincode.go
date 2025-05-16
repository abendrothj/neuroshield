package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

const (
	ChaincodeVersion = "1.0.1"
	EventTypeIndex   = "eventType~timestamp~id"
	MaxPageSize      = 100
)

type SmartContract struct {
	contractapi.Contract
}

type SecurityEvent struct {
	ID          string `json:"id"`
	Timestamp   string `json:"timestamp"`
	Type        string `json:"type"`
	Details     string `json:"details"`
	IPFSHash    string `json:"ipfshash"`
	EventHash   string `json:"eventHash"`
	TxID        string `json:"txId"`
	BlockNumber uint64 `json:"blockNumber"`
	Version     string `json:"version"`
}

type TransactionReceipt struct {
	TxID        string `json:"txId"`
	EventID     string `json:"eventId"`
	Timestamp   string `json:"timestamp"`
	BlockNumber uint64 `json:"blockNumber"`
	Status      string `json:"status"`
	EventHash   string `json:"eventHash"`
}

type QueryResult struct {
	Records    []*SecurityEvent `json:"records"`
	Bookmark   string           `json:"bookmark"`
	TotalCount int              `json:"totalCount"`
}

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	event := SecurityEvent{
		ID:          "init1",
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		Type:        "Initialization",
		Details:     "Ledger initialized for NeuraShield",
		IPFSHash:    "",
		EventHash:   "",
		TxID:        ctx.GetStub().GetTxID(),
		BlockNumber: 0,
		Version:     ChaincodeVersion,
	}
	eventJSON, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %v", err)
	}
	return ctx.GetStub().PutState("init1", eventJSON)
}

func (s *SmartContract) LogEvent(ctx contractapi.TransactionContextInterface, id, timestamp, eventType, details, ipfsHash, eventHash string) (*TransactionReceipt, error) {
	// Validate input
	if id == "" || timestamp == "" || eventType == "" {
		return nil, fmt.Errorf("required fields cannot be empty")
	}

	// Check if event already exists
	existingEvent, err := s.QueryEvent(ctx, id)
	if err == nil && existingEvent != nil {
		return nil, fmt.Errorf("event with ID %s already exists", id)
	}

	// Get transaction ID
	txID := ctx.GetStub().GetTxID()

	// Get current block height (will be incremented after this transaction)
	blockNumber, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return nil, fmt.Errorf("failed to get transaction timestamp: %v", err)
	}

	// Create the event
	event := SecurityEvent{
		ID:          id,
		Timestamp:   timestamp,
		Type:        eventType,
		Details:     details,
		IPFSHash:    ipfsHash,
		EventHash:   eventHash,
		TxID:        txID,
		BlockNumber: uint64(blockNumber.Seconds), // Using seconds as a rough proxy for block number
		Version:     ChaincodeVersion,
	}

	eventJSON, err := json.Marshal(event)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal event: %v", err)
	}

	// Store the event
	err = ctx.GetStub().PutState(id, eventJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to put event: %v", err)
	}

	// Create and store composite key
	compositeKey, err := ctx.GetStub().CreateCompositeKey(EventTypeIndex, []string{eventType, timestamp, id})
	if err != nil {
		return nil, fmt.Errorf("failed to create composite key: %v", err)
	}

	err = ctx.GetStub().PutState(compositeKey, []byte{0x00})
	if err != nil {
		return nil, fmt.Errorf("failed to put composite key: %v", err)
	}

	// Create transaction receipt
	receipt := TransactionReceipt{
		TxID:        txID,
		EventID:     id,
		Timestamp:   timestamp,
		BlockNumber: event.BlockNumber,
		Status:      "COMMITTED",
		EventHash:   eventHash,
	}

	// Return transaction receipt
	return &receipt, nil
}

func (s *SmartContract) VerifyEvent(ctx contractapi.TransactionContextInterface, id, providedHash string) (bool, error) {
	// Get the event
	event, err := s.QueryEvent(ctx, id)
	if err != nil {
		return false, fmt.Errorf("failed to get event: %v", err)
	}

	// Compare the provided hash with the stored hash
	return event.EventHash == providedHash, nil
}

func (s *SmartContract) QueryEvent(ctx contractapi.TransactionContextInterface, id string) (*SecurityEvent, error) {
	eventJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return nil, fmt.Errorf("failed to read from ledger: %v", err)
	}
	if eventJSON == nil {
		return nil, fmt.Errorf("event %s does not exist", id)
	}

	var event SecurityEvent
	err = json.Unmarshal(eventJSON, &event)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal event: %v", err)
	}

	return &event, nil
}

func (s *SmartContract) QueryEventsByType(ctx contractapi.TransactionContextInterface, eventType string, pageSize int, bookmark string) (*QueryResult, error) {
	if pageSize <= 0 || pageSize > MaxPageSize {
		pageSize = MaxPageSize
	}

	resultsIterator, metadata, err := ctx.GetStub().GetStateByPartialCompositeKeyWithPagination(EventTypeIndex, []string{eventType}, int32(pageSize), bookmark)
	if err != nil {
		return nil, fmt.Errorf("failed to get events by type: %v", err)
	}
	defer resultsIterator.Close()

	var events []*SecurityEvent
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, fmt.Errorf("failed to get next result: %v", err)
		}

		_, compositeKeyParts, err := ctx.GetStub().SplitCompositeKey(queryResponse.Key)
		if err != nil {
			return nil, fmt.Errorf("failed to split composite key: %v", err)
		}

		eventID := compositeKeyParts[2]
		event, err := s.QueryEvent(ctx, eventID)
		if err != nil {
			return nil, fmt.Errorf("failed to get event: %v", err)
		}

		events = append(events, event)
	}

	return &QueryResult{
		Records:    events,
		Bookmark:   metadata.GetBookmark(),
		TotalCount: len(events),
	}, nil
}

func (s *SmartContract) QueryEventsWithPagination(ctx contractapi.TransactionContextInterface, pageSize int, bookmark string) (*QueryResult, error) {
	if pageSize <= 0 || pageSize > MaxPageSize {
		pageSize = MaxPageSize
	}

	resultsIterator, metadata, err := ctx.GetStub().GetStateByRangeWithPagination("", "", int32(pageSize), bookmark)
	if err != nil {
		return nil, fmt.Errorf("failed to get events with pagination: %v", err)
	}
	defer resultsIterator.Close()

	var events []*SecurityEvent
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, fmt.Errorf("failed to get next result: %v", err)
		}

		// Skip composite keys
		if len(queryResponse.Value) == 1 && queryResponse.Value[0] == 0x00 {
			continue
		}

		var event SecurityEvent
		err = json.Unmarshal(queryResponse.Value, &event)
		if err != nil {
			continue // Skip invalid entries
		}

		events = append(events, &event)
	}

	return &QueryResult{
		Records:    events,
		Bookmark:   metadata.GetBookmark(),
		TotalCount: len(events),
	}, nil
}

func (s *SmartContract) QueryTransactionReceipt(ctx contractapi.TransactionContextInterface, txID string) (*TransactionReceipt, error) {
	// Query all events
	result, err := s.QueryEventsWithPagination(ctx, MaxPageSize, "")
	if err != nil {
		return nil, fmt.Errorf("failed to query events: %v", err)
	}

	// Find event with matching transaction ID
	for _, event := range result.Records {
		if event.TxID == txID {
			receipt := TransactionReceipt{
				TxID:        event.TxID,
				EventID:     event.ID,
				Timestamp:   event.Timestamp,
				BlockNumber: event.BlockNumber,
				Status:      "COMMITTED",
				EventHash:   event.EventHash,
			}
			return &receipt, nil
		}
	}

	return nil, fmt.Errorf("transaction with ID %s not found", txID)
}

func (s *SmartContract) GetVersion(ctx contractapi.TransactionContextInterface) (string, error) {
	return ChaincodeVersion, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		fmt.Printf("Error creating chaincode: %v\n", err)
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting chaincode: %v\n", err)
	}
}

package main

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type SmartContract struct {
	contractapi.Contract
}

type SecurityEvent struct {
	ID        string `json:"id"`
	Timestamp string `json:"timestamp"`
	Type      string `json:"type"`
	Details   string `json:"details"`
	IPFSHash  string `json:"ipfshash"` // IPFS Content Identifier for full logs
}

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	event := SecurityEvent{
		ID:        "init1",
		Timestamp: "2025-04-03T12:00:00Z",
		Type:      "Initialization",
		Details:   "Ledger initialized for NeuraShield",
		IPFSHash:  "", // No IPFS hash for initialization
	}
	eventJSON, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState("init1", eventJSON)
}

func (s *SmartContract) LogEvent(ctx contractapi.TransactionContextInterface, id, timestamp, eventType, details, ipfsHash string) error {
	event := SecurityEvent{
		ID:        id,
		Timestamp: timestamp,
		Type:      eventType,
		Details:   details,
		IPFSHash:  ipfsHash,
	}
	eventJSON, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(id, eventJSON)
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
		return nil, err
	}
	return &event, nil
}

func (s *SmartContract) QueryAllEvents(ctx contractapi.TransactionContextInterface) ([]*SecurityEvent, error) {
	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var events []*SecurityEvent
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}
		var event SecurityEvent
		err = json.Unmarshal(queryResponse.Value, &event)
		if err != nil {
			return nil, err
		}
		events = append(events, &event)
	}
	return events, nil
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

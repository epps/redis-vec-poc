package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
	"github.com/redis/rueidis"
)

// Query to test whether FLOAT64 is supported, which is wasn't on Redis 6.2.6
// FT.CREATE idx SCHEMA quote_vector VECTOR HNSW 2 TYPE FLOAT64

var modelId string = "sentence-transformers/all-MiniLM-L6-v2"
var hg_api string = fmt.Sprintf("https://api-inference.huggingface.co/pipeline/feature-extraction/%s", modelId)

type FeatureExtractionOptions struct {
	UseCache     bool `json:"use_cache"`
	WaitForModel bool `json:"wait_for_model"`
}

type FeatureExtractionPayload struct {
	Inputs  string                   `json:"inputs"`
	Options FeatureExtractionOptions `json:"options"`
}

type FeatureExtractionResponse []float64

type PointBreakQuote struct {
	Character string
	Quote     string
	Vector    []float64
}

func main() {
	var loadData bool
	var query string
	flag.BoolVar(&loadData, "load", false, "Load data into Redis")
	flag.StringVar(&query, "query", "Surfing bank robbers", "Query to search")
	flag.Parse()

	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	client, err := rueidis.NewClient(rueidis.ClientOption{
		InitAddress: []string{"localhost:6379"},
	})
	if err != nil {
		log.Fatalf("Failed to create Redis client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	if loadData {
		loadQuoteData(client, ctx)
	}

	// TEST SEARCH
	vec, err := vectorizeText(query)
	if err != nil {
		log.Fatalf("Failed to vectorize text: %v", err)
	}

	t, docs, err := kNearest(client, ctx, vec)
	if err != nil {
		log.Fatalf("KNN search failed: %v", err)
	}

	fmt.Println("Total:", t)
	fmt.Println("Docs:", docs)

	fmt.Println("Complete")
}

func vectorizeText(inputs string) (FeatureExtractionResponse, error) {
	payload := FeatureExtractionPayload{
		Inputs: inputs,
		Options: FeatureExtractionOptions{
			UseCache:     true,
			WaitForModel: true,
		},
	}

	jsonStr, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("error marshalling payload: %v", err)
	}

	req, err := http.NewRequest("POST", hg_api, bytes.NewBuffer(jsonStr))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", os.Getenv("HG_TOKEN")))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	fmt.Println("Hugging Face Feature Extraction Response Status:", resp.Status)

	var data FeatureExtractionResponse
	err = json.NewDecoder(resp.Body).Decode(&data)
	if err != nil {
		return nil, fmt.Errorf("error decoding response body: %v", err)
	}

	return data, nil
}

func buildHsetCommands(client rueidis.Client) (rueidis.Commands, error) {
	cmds := make(rueidis.Commands, len(quotes))
	for i, quote := range quotes {
		vector, err := vectorizeText(quote.Quote)
		if err != nil {
			return nil, err
		}

		key := fmt.Sprintf("quotes:%d", i)
		cmds[i] = client.B().
			Hset().Key(key).FieldValue().
			FieldValue("character", quote.Character).
			FieldValue("quote", quote.Quote).
			FieldValue("quote_vector", rueidis.VectorString64(vector)).
			Build()
	}

	return cmds, nil
}

// See https://redis.io/docs/stack/search/reference/vectors/#hnsw
func buildHnswIndex(client rueidis.Client) rueidis.Completed {
	idx := client.B().FtCreate().Index("idx")
	args := []string{
		// "TYPE", "FLOAT32", // Use float32 with Redis 6.2.6
		"TYPE", "FLOAT64",
		"DIM", "384", // Determined by model used to create vectors
		"DISTANCE_METRIC", "COSINE", // Choices are COSINE, L2, IP
		"INITIAL_CAP", "100", // Initial capacity of the index
		"M", "40",
		"EF_CONSTRUCTION", "200",
	}
	schema := idx.Schema().FieldName("quote_vector").Vector("HNSW", int64(len(args)), args...)
	b := schema.Build()
	fmt.Println("INDEX COMMANDS => ", b.Commands())
	return b
}

func kNearest(client rueidis.Client, ctx context.Context, vec []float64) (int64, []rueidis.FtSearchDoc, error) {
	query := client.B().FtSearch().
		Index("idx").
		Query("*=>[KNN $K @quote_vector $BLOB]").
		/*
			It's unclear why I need to specify a number for returned docs if the query already
			determines the return count; additionally, it seems that the `Identifier` methods
			must chained from the `Return` method, otherwise the query won't compile.
		*/
		// Return("10").
		// Identifier("character").
		// Identifier("quote").
		Sortby("__quote_vector_score").
		Params().
		Nargs(4).
		NameValue().
		NameValue("K", "10").
		NameValue("BLOB", rueidis.VectorString64(vec)).
		Dialect(2).
		Build()

	// fmt.Println("Query:", query.Commands())

	return client.Do(ctx, query).AsFtSearch()
}

func loadQuoteData(client rueidis.Client, ctx context.Context) error {
	cmds := make(rueidis.Commands, len(quotes)+1)
	cmds[0] = buildHnswIndex(client)
	hsetCmds, err := buildHsetCommands(client)
	if err != nil {
		return err
	}
	for i, cmd := range hsetCmds {
		cmds[i+1] = cmd
	}

	for _, resp := range client.DoMulti(ctx, cmds...) {
		if err := resp.Error(); err != nil {
			return err
		}
	}

	return nil
}

package com.example.flashcard_project.async_tasks.inputs;

import com.example.flashcard_project.beans.FlashCard;

import java.util.List;

public class CardAsyncInputs {

    private List<FlashCard> cards;
    private List<String> fcids;
    private FlashCard card;
    private String userId;
    private String userToken;
    private byte level;
    private int offset;

    public CardAsyncInputs(List<FlashCard> cards, FlashCard card, String userId, String userToken) {
        this.cards = cards;
        this.card = card;
        this.userId = userId;
        this.userToken = userToken;
    }

    public CardAsyncInputs(List<String> fcids) {
        this.fcids = fcids;
    }

    public CardAsyncInputs(String userId, byte level) {
        this.cards = cards;
        this.card = card;
        this.userId = userId;
        this.userToken = userToken;
        this.level = level;
    }

    public CardAsyncInputs(int offset) {
        this.offset = offset;
    }

    public CardAsyncInputs() {
    }

    public List<FlashCard> getCards() {
        return cards;
    }

    public void setCards(List<FlashCard> cards) {
        this.cards = cards;
    }

    public FlashCard getCard() {
        return card;
    }

    public void setCard(FlashCard card) {
        this.card = card;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getUserToken() {
        return userToken;
    }

    public void setUserToken(String userToken) {
        this.userToken = userToken;
    }

    public byte getLevel() {
        return level;
    }

    public void setLevel(byte level) {
        this.level = level;
    }

    public List<String> getFcids() {
        return fcids;
    }

    public void setFcids(List<String> fcids) {
        this.fcids = fcids;
    }

    public int getOffset() {
        return offset;
    }

    public void setOffset(int offset) {
        this.offset = offset;
    }
}

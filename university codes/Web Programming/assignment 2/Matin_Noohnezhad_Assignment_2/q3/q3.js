var up = true;
var index = 1;
var correctSpace = false;

function updown(bool) {
    // window.alert("first");
    up = bool;
	correctSpace = true;
}

function indexSelector(input) {

    // window.alert("second");
    if(correctSpace){
	index = input;
    mainFunc();
	}
}

function mainFunc() {
	correctSpace = false;
    if (!((index == 1 && up == true) || (index == 10 && up == false))) {
        if (up == true) {
            var first = "i" + (index - 1).toString();
            var second = "i" + (index).toString();
            innerChanger(first, second);
        } else {
            var first = "i" + (index).toString();
            var second = "i" + (index + 1).toString();
            innerChanger(first, second);
        }
    }
}

function innerChanger(firstID, secondID) {
    var first = document.getElementById(firstID).innerHTML;
    var second = document.getElementById(secondID).innerHTML;
    document.getElementById(firstID).innerHTML = second;
    document.getElementById(secondID).innerHTML = first;
}

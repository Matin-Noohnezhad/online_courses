var colors = ["blue","yellow","red","green","pink","gray"];

function myFunc(){
    var color = document.getElementById("i").value.toLowerCase();
    var num = Math.floor(Math.random()*6);
    if(colors[num] == color){
        document.getElementById("b").style.backgroundColor = color;
    }
}
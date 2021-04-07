
var deleteStatus = false;

function formValidation(id) {
    id = "myForm"+id;
    var error = "error"+id;
    if(deleteStatus){
        return true;
    }
    var newName = document.forms[id]["newName"].value;
    var newPrice = document.forms[id]["newPrice"].value;

    if(newName.length==0 &&newPrice.length==0){
document.getElementById(error).innerHTML = 'برای اعمال تغییرات حداقل یک فیلد باید پر شود هر دو فیلد خالی است.*'
        return false;
    }
    return true

}

function setDelete(id) {
    id = "myForm"+id;
    document.forms[id]["delete"].value = 1;
    deleteStatus = true;
}

